//! Bounded LRU checkpoints of complete hybrid-model sequence state.

use crate::{
    options::Options,
    qwen4_exp::gpu::{Gpu, MAX_NB, PrefixState},
    runner::PrefillResume,
};
use anyhow::Result;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

struct Entry {
    tokens: Vec<u32>,
    following: Option<u32>,
    complete: bool,
    next: u32,
    drafts: Vec<u32>,
    state: PrefixState,
    bytes: usize,
    touched: Instant,
}

pub(crate) struct PrefixCache {
    entries: VecDeque<Entry>,
    capacity: usize,
    used: usize,
    evictions: u64,
    max_entries: usize,
    idle: Option<Duration>,
}

fn matches(tokens: &[u32], following: Option<u32>, complete: bool, request: &[u32]) -> bool {
    request.starts_with(tokens)
        && if request.len() == tokens.len() {
            complete
        } else {
            // The last MTP cache row depends on the token AFTER the trunk prefix.
            // Never reuse it for a different continuation.
            following.is_none_or(|next| request[tokens.len()] == next)
        }
}

impl PrefixCache {
    pub(crate) fn new(capacity: usize, max_entries: usize, idle_seconds: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            used: 0,
            evictions: 0,
            max_entries,
            idle: (idle_seconds > 0).then(|| Duration::from_secs(idle_seconds)),
        }
    }

    fn evict_oldest(&mut self) {
        if let Some(entry) = self.entries.pop_front() {
            self.used -= entry.bytes;
            self.evictions += 1;
        }
    }

    pub(crate) fn expire(&mut self) {
        if let Some(idle) = self.idle {
            while self
                .entries
                .front()
                .is_some_and(|e| e.touched.elapsed() >= idle)
            {
                self.evict_oldest();
            }
        }
    }

    pub(crate) fn stats(&self) -> (usize, usize, u64) {
        (self.entries.len(), self.used, self.evictions)
    }

    fn store(&mut self, gpu: &Gpu<'_>, ids: &[u32], next: u32, drafts: &[u32]) {
        let pos = gpu.pos;
        let bytes =
            gpu.prefix_state_bytes() + pos * 4 + std::mem::size_of::<Entry>() + drafts.len() * 4;
        if bytes > self.capacity || pos == 0 {
            return;
        }

        if let Some(i) = self.entries.iter().position(|e| e.tokens == ids[..pos]) {
            self.used -= self.entries.remove(i).unwrap().bytes;
        }
        // Evict BEFORE copying state: allocation peaks also stay within the cache budget.
        while self.used + bytes > self.capacity || self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        let following = gpu.has_mtp().then(|| ids.get(pos).copied().unwrap_or(next));
        self.entries.push_back(Entry {
            tokens: ids[..pos].to_vec(),
            following,
            complete: pos == ids.len(),
            next,
            drafts: drafts.to_vec(),
            state: gpu.save_prefix(),
            bytes,
            touched: Instant::now(),
        });
        self.used += bytes;
    }

    pub(crate) fn prepare(
        &mut self,
        gpu: &mut Gpu<'_>,
        ids: &[u32],
        stable_boundaries: &[usize],
        options: &Options,
    ) -> Result<(PrefillResume, usize)> {
        let started = std::time::Instant::now();
        self.expire();

        let mut cached = 0;
        if let Some(i) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| matches(&e.tokens, e.following, e.complete, ids))
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)
        {
            let mut entry = self.entries.remove(i).unwrap();
            entry.touched = Instant::now();
            gpu.restore_prefix(&entry.state)?;
            cached = entry.tokens.len();
            let seed = PrefillResume {
                next: entry.next,
                drafts: entry.drafts.clone(),
            };
            self.entries.push_back(entry);
            if cached == ids.len() {
                eprintln!(
                    "prefix cache: {cached}/{} tokens reused, {:.1} MiB cached",
                    ids.len(),
                    self.used as f64 / 1048576.0
                );
                return Ok((seed, cached));
            }
        }

        let pf_min = std::env::var("CHERENKOV_PREFILL_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let pf_rows = std::env::var("CHERENKOV_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| gpu.prefill_rows_fit())
            .max(1);
        let row_max = std::env::var("CHERENKOV_ROWS_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MAX_NB)
            .clamp(1, MAX_NB);

        let mut boundaries = vec![ids.len()];
        if self.capacity > 0 {
            for &boundary in stable_boundaries {
                if boundary > cached && boundary < ids.len() {
                    boundaries.push(boundary);
                }
            }
            // This checkpoint supports arbitrary prompt extensions: its MTP
            // lookahead is the old prompt's known final token, not a prediction.
            if ids.len() > 1 {
                boundaries.push(ids.len() - 1);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();

        let mut seed = PrefillResume {
            next: ids[0],
            drafts: Vec::new(),
        };
        for end in boundaries {
            if end <= gpu.pos {
                continue;
            }
            let remaining = end - gpu.pos;
            let engine = remaining >= pf_min;
            let chunk_size = if engine {
                remaining.div_ceil(remaining.div_ceil(pf_rows))
            } else {
                row_max
            };
            while gpu.pos < end {
                let pos = gpu.pos;
                let n = (end - pos).min(chunk_size);
                let rows = &ids[pos..pos + n];
                let following = ids.get(pos + n).copied();
                seed = prefill_rows(gpu, rows, following, options.drafts as usize, engine)?;
                if gpu.pos == end || engine {
                    self.store(gpu, ids, seed.next, &seed.drafts);
                }
            }
            gpu.prefill_release();
        }

        eprintln!(
            "prefix cache: {cached}/{} tokens reused, {} prefetched in {:.3}s, {:.1} MiB cached",
            ids.len(),
            ids.len() - cached,
            started.elapsed().as_secs_f64(),
            self.used as f64 / 1048576.0
        );
        Ok((seed, cached))
    }
}

/// Advance one prompt chunk and retain the predictions needed to resume decode.
fn prefill_rows(
    gpu: &mut Gpu<'_>,
    rows: &[u32],
    following: Option<u32>,
    drafts: usize,
    engine: bool,
) -> Result<PrefillResume> {
    let last = following.is_none();
    if engine {
        let (next, draft) = gpu.prefill_chunk(rows, following, false)?;
        let mut seed = PrefillResume {
            next,
            drafts: Vec::new(),
        };
        if drafts == 0 || !last {
            return Ok(seed);
        }
        seed.drafts.push(draft);
        if drafts >= 2 {
            seed.drafts.push(gpu.mtp_chain(draft)?);
        }
        return Ok(seed);
    }

    let result = gpu.step_rows(rows, false, false)?;
    gpu.commit(rows.len())?;
    let mut seed = PrefillResume {
        next: result[rows.len() - 1],
        drafts: Vec::new(),
    };
    if drafts == 0 {
        return Ok(seed);
    }

    let mut next = rows[1..].to_vec();
    next.push(following.unwrap_or(seed.next));
    seed.drafts = gpu.mtp_draft(&next, if last { drafts } else { 1 })?;
    Ok(seed)
}

#[cfg(test)]
#[path = "../tests/unit/prefix_cache.rs"]
mod tests;
