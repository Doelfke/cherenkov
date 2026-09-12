//! Cumulative completed prefill chunks. Sequence rollback does not undo work.

use super::ChunkStats;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct PrefillStats {
    pub chunks: u64,
    pub tokens: u64,
    pub min_chunk_tokens: usize,
    pub max_chunk_tokens: usize,
    pub seconds: f64,
    pub ring_wait_seconds: f64,
    pub gpu_delta_seconds: f64,
    pub gpu_attention_seconds: f64,
    pub gpu_expert_seconds: f64,
    pub gpu_mtp_seconds: f64,
    pub ngram_seconds: f64,
}

impl PrefillStats {
    pub(super) fn record(&mut self, chunk: ChunkStats) {
        self.min_chunk_tokens = if self.chunks == 0 {
            chunk.tokens
        } else {
            self.min_chunk_tokens.min(chunk.tokens)
        };
        self.max_chunk_tokens = self.max_chunk_tokens.max(chunk.tokens);
        self.chunks += 1;
        self.tokens += chunk.tokens as u64;
        self.seconds += chunk.secs;
        self.ring_wait_seconds += chunk.wait_s;
        self.gpu_delta_seconds += chunk.gpu_delta_s;
        self.gpu_attention_seconds += chunk.gpu_attn_s;
        self.gpu_expert_seconds += chunk.gpu_experts_s;
        self.gpu_mtp_seconds += chunk.gpu_mtp_s;
        self.ngram_seconds += chunk.ngram_s;
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/qwen4_exp/gpu/prefill_stats.rs"]
mod tests;
