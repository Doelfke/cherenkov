//! The serial inference worker owns GPU state and accounts for each job once.

use super::{Job, PreparedRequest, error, parse_request, response::Response, sse};
use crate::{
    control::State, options::Options, prefix_cache::PrefixCache, qwen4_exp::gpu::Gpu, runner,
    tok::ChatTokenizer,
};
use anyhow::Result;
use serde_json::json;
use std::{
    io::Write,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

pub(super) struct Worker<'a> {
    gpu: Gpu<'a>,
    tok: ChatTokenizer,
    options: Options,
    cache: PrefixCache,
    state: Arc<State>,
    serial: u64,
}

impl<'a> Worker<'a> {
    pub(super) fn new(
        gpu: Gpu<'a>,
        tok: ChatTokenizer,
        options: Options,
        cache: PrefixCache,
        state: Arc<State>,
    ) -> Self {
        Self {
            gpu,
            tok,
            options,
            cache,
            state,
            serial: 0,
        }
    }

    pub(super) fn expire_cache(&mut self) {
        self.cache.expire();
        self.state.observe(&self.gpu, &self.cache);
    }

    pub(super) fn handle(&mut self, mut job: Job) {
        self.state.begin(job.settings.generation);
        let prepared =
            parse_request(&job.body, job.kind, &job.settings.config.defaults).and_then(|r| {
                r.prepare(
                    &self.tok,
                    &self.options,
                    job.settings.config.limits.max_output_tokens,
                )
            });
        let result = match prepared {
            Err(e) => {
                let _ = error(&mut job.stream, 400, &e.to_string());
                Err(e)
            }
            Ok(prepared) => {
                self.serial += 1;
                self.gpu.reset_request();
                let result = self.complete(&mut job, &prepared);
                if let Err(e) = &result {
                    if prepared.request.stream {
                        let _ = sse(
                            &mut job.stream,
                            &json!({"error":{"message":e.to_string(),"type":"server_error"}}),
                        );
                        let _ = job.stream.write_all(b"data: [DONE]\n\n");
                    } else {
                        let _ = error(&mut job.stream, 500, &e.to_string());
                    }
                }
                result
            }
        };
        self.state.observe(&self.gpu, &self.cache);
        self.state.finish(result.is_ok());
        if let Err(e) = result {
            eprintln!("request failed: {e:#}");
        }
    }

    fn complete(&mut self, job: &mut Job, prepared: &PreparedRequest) -> Result<()> {
        let PreparedRequest {
            request,
            ids,
            boundaries,
            options,
        } = prepared;
        let Self {
            gpu,
            tok,
            cache,
            state,
            serial,
            ..
        } = self;
        let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let mut response = Response::new(&mut job.stream, job.kind, *serial, created, request);
        response.start()?;
        state.phase("prefill");
        let prefill_start = std::time::Instant::now();
        let prepared = cache.prepare(gpu, ids, boundaries, options);
        state.update(|s| s.prefill_seconds += prefill_start.elapsed().as_secs_f64());
        let (resume, cached_tokens) = prepared?;
        state.update(|s| {
            s.prompt_tokens += ids.len() as u64;
            s.cached_tokens += cached_tokens as u64;
        });
        state.observe(gpu, cache);
        state.phase("decode");
        let decode_start = std::time::Instant::now();
        let mut decoder = tok.inner.decode_stream(false);
        let mut text = String::new();
        let result = runner::qwen4_exp_gen_once(
            gpu,
            None,
            None,
            tok,
            ids,
            request.max_tokens,
            options,
            Some(resume),
            &mut |token| {
                state.token();
                if let Some(delta) = decoder
                    .step(token)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))?
                {
                    text.push_str(&delta);
                    response.text(&delta)?;
                }
                Ok(())
            },
        );
        state.update(|s| s.decode_seconds += decode_start.elapsed().as_secs_f64());
        let result = result?;
        // Complete a trailing partial Unicode sequence consistently for both response modes.
        let full = tok.decode(&result.tokens)?;
        if let Some(tail) = full.strip_prefix(&text).filter(|s| !s.is_empty()) {
            response.text(tail)?;
            text.push_str(tail);
        }
        let usage = json!({
            "prompt_tokens": result.prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens": result.tokens.len(),
            "total_tokens": result.prompt_tokens + result.tokens.len(),
        });
        response.finish(&text, result.finish_reason, usage)
    }
}
