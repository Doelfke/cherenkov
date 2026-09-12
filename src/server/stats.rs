//! Generation work, shared by active-request and committed-session snapshots.

use serde::Serialize;

#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

impl UsageStats {
    pub(super) fn add(&mut self, other: Self) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.cached_tokens = self.cached_tokens.saturating_add(other.cached_tokens);
        self.generated_tokens = self.generated_tokens.saturating_add(other.generated_tokens);
        self.prefill_seconds += other.prefill_seconds;
        self.decode_seconds += other.decode_seconds;
    }
}
