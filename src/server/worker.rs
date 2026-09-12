//! Round-robin GPU scheduling. Each active request owns its decoding state.

use super::{
    Job, UsageStats,
    http::error,
    output::{Frame, Output},
    pacer::ChunkPacer,
    registry::Ticket,
    request::{PreparedRequest, parse_request},
    sessions::{Store, Turn},
};
use crate::units::BYTES_PER_MIB;
use crate::{
    control::{State, state::ActiveRequest},
    options::Options,
    prefix_cache::{Prefill, PrefixCache},
    qwen4_exp::gpu::{Gpu, PrefixState},
    runner::Decode,
    tok::ChatTokenizer,
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

struct Pending {
    job: Job,
    prepared: PreparedRequest,
    reservation: usize,
}

enum Phase {
    Ready,
    Prefill(Prefill),
    Decode(Box<Decode>),
}

struct Active<'a> {
    prepared: PreparedRequest,
    phase: Phase,
    checkpoint: Option<PrefixState>,
    reservation: usize,
    ticket: Arc<Ticket>,
    session: Option<Turn>,
    output: Output,
    text_decoder: Box<dyn FnMut(u32) -> Result<Option<String>> + 'a>,
    text: String,
    usage: UsageStats,
    config_generation: u64,
    response_bytes: usize,
    /// Tokens and elapsed seconds from the last step, if eligible for pacing.
    last_chunk: Option<(usize, f64)>,
}

pub(super) struct Worker<'a> {
    activity_snapshot: crate::qwen4_exp::gpu::ExpertActivity,
    gpu: Gpu<'a>,
    tok: &'a ChatTokenizer,
    options: Options,
    cache: PrefixCache,
    state: Arc<State>,
    sessions: Arc<Mutex<Store>>,
    active: VecDeque<Active<'a>>,
    pending: VecDeque<Pending>,
    loaded: Option<String>,
    reserved: usize,
    pacer: ChunkPacer,
}

impl<'a> Worker<'a> {
    pub(super) fn new(
        gpu: Gpu<'a>,
        tok: &'a ChatTokenizer,
        options: Options,
        cache: PrefixCache,
        state: Arc<State>,
        sessions: Arc<Mutex<Store>>,
    ) -> Self {
        let limits = &state.config().config.limits;
        let pacer = ChunkPacer::new(limits.prefill_quantum, limits.prefill_chunk_seconds);

        Self {
            activity_snapshot: Default::default(),
            gpu,
            tok,
            options,
            cache,
            state,
            sessions,
            active: VecDeque::new(),
            pending: VecDeque::new(),
            loaded: None,
            reserved: 0,
            pacer,
        }
    }

    pub(super) fn run(mut self, receiver: mpsc::Receiver<Job>) {
        loop {
            if self.active.is_empty() && self.pending.is_empty() {
                match receiver.recv_timeout(Duration::from_secs(1)) {
                    Ok(job) => self.prepare(job),
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }

            for job in receiver.try_iter() {
                self.prepare(job);
            }

            self.admit();
            self.turn();
            self.gpu.clear_profile();
            self.cache.expire();
            self.observe();
        }
    }

    fn prepare(&mut self, mut job: Job) {
        let result = parse_request(
            &job.body,
            job.kind,
            &job.settings.config.defaults,
            job.session.as_ref().map(|turn| &turn.input),
            self.tok.template.as_ref(),
        )
        .and_then(|r| {
            r.prepare(
                self.tok,
                &self.options,
                job.settings.config.limits.max_output_tokens,
            )
        });
        let prepared = match result {
            Ok(p) => p,
            Err(e) => {
                let _ = error(&mut job.stream, 400, &e.to_string());

                self.state.update(|s| {
                    s.queued_requests -= 1;
                    s.failed_requests += 1;
                });

                return;
            }
        };
        let limits = &job.settings.config.limits;
        let context =
            prepared.ids.len() + prepared.request.max_tokens + prepared.options.effective_drafts();
        // Vec growth can double initialized snapshot bytes. Reserve that upper
        // bound, sampler workspace, tokens, and bounded request/response copies.
        let draft_context = if self.gpu.has_mtp() { context } else { 0 };
        let checkpoint_bytes = 2 * self.gpu.state_bytes_at(context, draft_context);
        let sampler_bytes = self.gpu.logits().len() * 24;
        let token_bytes = context * 16;
        let message_bytes = 4 * (limits.request_bytes + limits.response_bytes);
        let reservation = checkpoint_bytes + sampler_bytes + token_bytes + message_bytes;

        if reservation > limits.active_state_mib * BYTES_PER_MIB {
            let _ = error(
                &mut job.stream,
                503,
                "request exceeds active_state_mib; reduce token limits or increase the state budget",
            );

            self.state.update(|s| {
                s.queued_requests -= 1;
                s.rejected_requests += 1;
            });

            return;
        }

        job.body = serde_json::Value::Null;

        self.pending.push_back(Pending {
            job,
            prepared,
            reservation,
        });
    }

    fn admit(&mut self) {
        let mut waiting = VecDeque::new();

        while let Some(mut pending) = self.pending.pop_front() {
            if pending.job.ticket.cancelled() {
                let _ = error(
                    &mut pending.job.stream,
                    499,
                    "request cancelled before admission",
                );

                self.state.update(|s| {
                    s.queued_requests -= 1;
                    s.cancelled_requests += 1;
                });

                continue;
            }

            let limits = &pending.job.settings.config.limits;

            if self.active.len() >= limits.active_requests
                || self.reserved + pending.reservation > limits.active_state_mib * BYTES_PER_MIB
            {
                waiting.push_back(pending);

                continue;
            }

            self.state.begin();

            self.reserved += pending.reservation;
            let mut decoder = self.tok.inner.decode_stream(false);
            let output = Output::new(
                pending.job.stream,
                pending.job.kind,
                &pending.prepared.request,
                pending.job.ticket.clone(),
                self.state.clone(),
            );

            self.active.push_back(Active {
                prepared: pending.prepared,
                phase: Phase::Ready,
                checkpoint: None,
                reservation: pending.reservation,
                ticket: pending.job.ticket,
                session: pending.job.session,
                output,
                text_decoder: Box::new(move |token| {
                    decoder
                        .step(token)
                        .map_err(|e| anyhow::anyhow!("decode: {e}"))
                }),
                text: String::new(),
                usage: UsageStats::default(),
                config_generation: pending.job.settings.generation,
                response_bytes: limits.response_bytes,
                last_chunk: None,
            });
        }

        self.pending = waiting;

        self.state
            .update(|s| s.active_state_reserved_bytes = self.reserved);
    }

    fn activate(&mut self, request: &Active<'_>) -> Result<()> {
        if self.loaded.as_deref() == Some(&request.ticket.id) {
            return Ok(());
        }

        if let Some(previous) = self
            .active
            .iter_mut()
            .find(|a| Some(&a.ticket.id) == self.loaded.as_ref())
        {
            self.gpu.save_into(&mut previous.checkpoint);
        }

        self.gpu.reset_request();

        if let Some(checkpoint) = &request.checkpoint {
            self.gpu.restore_prefix(checkpoint)?;
        }

        self.loaded = Some(request.ticket.id.clone());

        Ok(())
    }

    fn turn(&mut self) {
        let Some(mut request) = self.active.pop_front() else {
            return;
        };
        // Both paced and lone-request chunks stay within the reserved quantum.
        let contended = !self.active.is_empty() || !self.pending.is_empty();
        let quantum = self.pacer.quantum(contended);
        let result = if request.ticket.cancelled() {
            Ok(true)
        } else {
            self.activate(&request).and_then(|()| {
                request.step(
                    &mut self.gpu,
                    &mut self.cache,
                    self.tok,
                    &self.state,
                    quantum,
                )
            })
        };

        if let Some((tokens, seconds)) = request.last_chunk.take() {
            self.pacer.observe(tokens, seconds);
        }

        if matches!(result, Ok(false)) && !request.ticket.cancelled() {
            self.active.push_back(request);

            return;
        }

        self.reserved -= request.reservation;

        if self.loaded.as_deref() == Some(&request.ticket.id) {
            self.loaded = None;
        }

        let cancelled = request.ticket.cancelled();
        // The output writer accounts for completion and delivery failures.
        let _ = match result {
            Err(e) if !cancelled => request.output.send(Frame::Error(e.to_string())),
            _ => request.finish(self.tok),
        };

        self.state.update(|s| {
            s.active_requests -= 1;
            s.active_state_reserved_bytes = self.reserved;
            s.current = None;
        });
    }

    fn observe(&mut self) {
        self.state
            .observe(&self.gpu, &self.cache, &mut self.activity_snapshot);

        let mut sessions = self.sessions.lock().unwrap();

        sessions.expire();
        self.state.update(|s| {
            s.active_state_reserved_bytes = self.reserved;
            s.prefill_chunk_tokens = self.pacer.tokens();
            s.sessions = sessions.stats();
            s.active = self.active.iter().map(Active::stats).collect();
        });
    }
}

impl Active<'_> {
    fn stats(&self) -> ActiveRequest {
        ActiveRequest {
            id: self.ticket.id.clone(),
            session_id: self.session.as_ref().map(|s| s.id.clone()),
            phase: self.phase_name(),
            usage: self.usage,
            reserved_state_bytes: self.reservation,
        }
    }

    fn phase_name(&self) -> &'static str {
        match self.phase {
            Phase::Decode(_) => "decode",
            _ => "prefill",
        }
    }

    fn generated(&self) -> &[u32] {
        match &self.phase {
            Phase::Decode(d) => &d.tokens,
            _ => &[],
        }
    }

    fn step(
        &mut self,
        gpu: &mut Gpu<'_>,
        cache: &mut PrefixCache,
        tok: &ChatTokenizer,
        state: &State,
        quantum: usize,
    ) -> Result<bool> {
        state.current(
            &self.ticket.id,
            self.config_generation,
            self.phase_name(),
            self.generated().len(),
        );

        if matches!(self.phase, Phase::Ready) {
            let prefill = cache.begin(
                gpu,
                &self.prepared.ids,
                &self.prepared.boundaries,
                &self.prepared.options,
            )?;
            self.usage.prompt_tokens = self.prepared.ids.len() as u64;
            self.usage.cached_tokens = prefill.cached as u64;

            state.update(|s| {
                s.prompt_tokens += self.prepared.ids.len() as u64;
                s.cached_tokens += prefill.cached as u64;
            });

            self.phase = Phase::Prefill(prefill);
        }

        let started = Instant::now();

        if let Phase::Prefill(prefill) = &mut self.phase {
            if self.ticket.cancelled() {
                return Ok(false);
            }

            let result = prefill.advance(
                gpu,
                cache,
                &self.prepared.ids,
                &self.prepared.options,
                quantum,
            );
            let elapsed = started.elapsed().as_secs_f64();
            self.usage.prefill_seconds += elapsed;

            state.update(|s| s.prefill_seconds += elapsed);

            let progress = result?;
            self.last_chunk = progress.pacing_tokens.map(|tokens| (tokens, elapsed));

            if !progress.done || self.ticket.cancelled() {
                return Ok(false);
            }

            let seed = std::mem::take(&mut prefill.seed);
            let rng = self.session.as_ref().and_then(|s| s.rng.clone());
            self.phase = Phase::Decode(Box::new(Decode::new(
                seed,
                &self.prepared.ids,
                tok,
                &self.prepared.options,
                self.prepared.request.max_tokens,
                rng,
            )?));

            return Ok(false);
        }

        let Phase::Decode(decoder) = &mut self.phase else {
            unreachable!()
        };

        let result = decoder.step(gpu, None, None, &mut |token| {
            ensure!(!self.ticket.cancelled(), "request cancelled");

            if let Some(delta) = (self.text_decoder)(token)? {
                ensure!(
                    self.text.len() + delta.len() <= self.response_bytes,
                    "response exceeds response_bytes"
                );
                self.text.push_str(&delta);
                self.output.send(Frame::Text(delta))?;
            }

            state.token();

            Ok(())
        });
        let elapsed = started.elapsed().as_secs_f64();
        self.usage.decode_seconds += elapsed;
        self.usage.generated_tokens = decoder.tokens.len() as u64;

        state.update(|s| s.decode_seconds += elapsed);

        result?;

        Ok(decoder.finish_reason.is_some())
    }

    fn finish(mut self, tok: &ChatTokenizer) -> Result<()> {
        let result = self.finish_response(tok);

        if let Err(e) = &result {
            let _ = self.output.send(Frame::Error(e.to_string()));
        }

        result
    }

    fn finish_response(&mut self, tok: &ChatTokenizer) -> Result<()> {
        let cancelled = self.ticket.cancelled();
        let full = tok.decode(self.generated())?;

        ensure!(
            full.len() <= self.response_bytes,
            "response exceeds response_bytes"
        );

        if let Some(tail) = full
            .strip_prefix(&self.text)
            .filter(|tail| !tail.is_empty())
        {
            self.output.send(Frame::Text(tail.to_owned()))?;
        }

        let (reason, rng) = match &self.phase {
            Phase::Decode(d) => (d.finish_reason.unwrap_or("cancelled"), d.rng()),
            _ => ("cancelled", None),
        };
        let usage = json!({
            "prompt_tokens": self.prepared.ids.len(),
            "prompt_tokens_details": {"cached_tokens": self.usage.cached_tokens},
            "completion_tokens": self.generated().len(),
            "total_tokens": self.prepared.ids.len() + self.generated().len(),
        });
        let turn = if cancelled {
            None
        } else {
            self.session
                .take()
                .map(|t| t.prepare(&full, rng, self.usage))
                .transpose()?
        };

        self.output.send(Frame::Finish {
            text: full,
            reason,
            usage,
            turn: turn.map(Box::new),
        })
    }
}
