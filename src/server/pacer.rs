//! Wall-time pacing within the prefill capacity reserved at load.

pub(super) struct ChunkPacer {
    floor: usize,
    ceiling: usize,
    /// Zero uses the configured ceiling for every chunk.
    target_seconds: f64,
    tokens: usize,
    contended: bool,
}

impl ChunkPacer {
    pub(super) fn new(ceiling: usize, target_seconds: f64) -> Self {
        let floor = ceiling.min(128);

        Self {
            floor,
            ceiling,
            target_seconds,
            tokens: if target_seconds == 0.0 {
                ceiling
            } else {
                floor
            },
            contended: false,
        }
    }

    /// A new contention period starts small. Work already on the GPU still
    /// finishes its current chunk before another request can run.
    pub(super) fn quantum(&mut self, contended: bool) -> usize {
        if self.target_seconds == 0.0 {
            return self.ceiling;
        }

        if contended && !self.contended {
            self.tokens = self.floor;
        }

        self.contended = contended;

        if contended { self.tokens } else { self.ceiling }
    }

    /// Adjust the target from a successful engine chunk. The caller excludes
    /// short tails, capacity-limited chunks, and the small-row path.
    pub(super) fn observe(&mut self, tokens: usize, seconds: f64) {
        if !self.contended || tokens == 0 || !seconds.is_finite() || seconds <= 0.0 {
            return;
        }

        let scaled = tokens as f64 * self.target_seconds / seconds;
        let damped = scaled.clamp(self.tokens as f64 * 0.5, self.tokens as f64 * 2.0);

        self.tokens = (damped as usize).clamp(self.floor, self.ceiling);
    }

    /// Current contended chunk target, before memory and boundary limits.
    pub(super) fn tokens(&self) -> usize {
        self.tokens
    }
}

#[cfg(test)]
#[path = "../../tests/unit/server/pacer.rs"]
mod tests;
