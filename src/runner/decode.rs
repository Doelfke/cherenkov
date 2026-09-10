//! One complete verification step, resumable between shared-GPU turns.

use super::*;
use crate::sampling::Sampler;

pub(crate) struct Decode {
    cur: u32,
    drafts: Vec<u32>,
    n_draft: usize,
    sampler: Option<Sampler>,
    eos: Option<[u32; 2]>,
    limit: usize,
    pub tokens: Vec<u32>,
    pub finish_reason: Option<&'static str>,
    pub steps: usize,
    pub accepted: usize,
}

#[cfg(test)]
#[path = "../../tests/unit/runner/decode.rs"]
mod tests;

impl Decode {
    pub(crate) fn rng(&self) -> Option<rand::rngs::StdRng> {
        self.sampler.as_ref().map(Sampler::rng)
    }

    pub(crate) fn new(
        seed: PrefillResume,
        ids: &[u32],
        tok: &tok::ChatTokenizer,
        options: &Options,
        limit: usize,
        rng: Option<rand::rngs::StdRng>,
    ) -> Result<Self> {
        let mut sampler = if options.sampling.greedy() {
            None
        } else {
            Some(Sampler::new(
                &options.sampling,
                ids,
                seed.logits.as_ref().map_or(0, Vec::len),
            )?)
        };
        if let (Some(sampler), Some(rng)) = (&mut sampler, rng) {
            sampler.set_rng(rng);
        }
        let cur = match &mut sampler {
            Some(s) => s.sample(
                seed.logits
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("sampling needs final prompt logits"))?,
            )?,
            None => seed.next,
        };
        Ok(Self {
            cur,
            drafts: seed.drafts,
            n_draft: options.effective_drafts(),
            sampler,
            eos: (!options.no_eos).then_some([tok.im_end, tok.endoftext]),
            limit,
            tokens: Vec::new(),
            finish_reason: None,
            steps: 0,
            accepted: 0,
        })
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos.is_some_and(|ids| ids.contains(&token))
    }

    fn emit(&mut self, token: u32, emit: &mut dyn FnMut(u32) -> Result<()>) -> Result<()> {
        self.tokens.push(token);
        if let Some(sampler) = &mut self.sampler {
            sampler.accept(token)?;
        }
        emit(token)
    }

    /// Returns after committed trunk/MTP work; GPU scratch can then be reused.
    pub(crate) fn step(
        &mut self,
        gpu: &mut qwen4_exp::gpu::Gpu<'_>,
        cpu: Option<&qwen4_exp::cpu::CpuModel<'_>>,
        cpu_state: Option<&mut qwen4_exp::cpu::State>,
        emit: &mut dyn FnMut(u32) -> Result<()>,
    ) -> Result<()> {
        if self.finish_reason.is_some() {
            return Ok(());
        }
        if self.is_eos(self.cur) {
            self.finish_reason = Some("stop");
            return Ok(());
        }
        if self.tokens.len() >= self.limit {
            self.finish_reason = Some("length");
            return Ok(());
        }
        let mut rows = vec![self.cur];
        let remaining = self.limit - self.tokens.len() - 1;
        rows.extend(self.drafts.iter().take(self.n_draft.min(remaining)));
        self.emit(self.cur, emit)?;
        let pos = gpu.pos;
        let result = gpu.step_rows(&rows, rows.len() > 1, self.n_draft > 0)?;
        let mut n = 1;
        while n < rows.len() && result[n - 1] == rows[n] && !self.is_eos(rows[n]) {
            n += 1;
        }
        gpu.commit(n)?;
        self.steps += 1;
        self.accepted += n - 1;
        let mut next = rows[1..n].to_vec();
        next.push(result[n - 1]);
        if self.n_draft > 0 {
            let chain = if self.n_draft >= 2 && n < rows.len() {
                1
            } else {
                self.n_draft
            };
            self.drafts = gpu.mtp_draft(&next, chain)?;
        }
        if let (Some(model), Some(state)) = (cpu, cpu_state) {
            qwen4_exp_check_rows(
                model,
                state,
                gpu,
                &rows[..n],
                (self.n_draft > 0).then_some(&next[..]),
                pos,
                "decode",
                false,
            )?;
        }
        for &token in &rows[1..n] {
            self.emit(token, emit)?;
        }
        if self.tokens.len() == self.limit {
            self.finish_reason = Some("length");
            return Ok(());
        }
        self.cur = match &mut self.sampler {
            Some(sampler) => sampler.sample(gpu.logits_row(n - 1))?,
            None => result[n - 1],
        };
        Ok(())
    }
}
