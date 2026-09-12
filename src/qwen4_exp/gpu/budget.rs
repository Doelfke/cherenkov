//! Expert-pool sizing after fixed buffers and host reservations.

use super::*;

pub(super) struct PoolMemory {
    pub device: usize,
    pub fixed: usize,
    pub host: usize,
    pub allocation_limit: Option<usize>,
    pub prefill: usize,
}

impl PoolMemory {
    pub fn bytes(&self, request: PoolBudget) -> Result<usize> {
        let available = self
            .allocation_limit
            .unwrap_or(usize::MAX)
            .saturating_sub(self.fixed)
            .saturating_sub(self.prefill);

        if let PoolBudget::Gb(gb) = request {
            let bytes = gb * BYTES_PER_GB as f64;

            ensure!(
                bytes.is_finite() && bytes > 0.0 && bytes < usize::MAX as f64,
                "expert pool size is not representable in bytes"
            );

            let bytes = bytes as usize;

            ensure!(
                bytes <= available,
                "explicit expert pool exceeds server memory budget after fixed buffers and prefill scratch"
            );

            return Ok(bytes);
        }

        // Adaptive keeps the existing 6 GB margin on the measured Air, but
        // no longer forces an 8 GB minimum or a 20 GB maximum on other Macs.
        let margin = if matches!(request, PoolBudget::Adaptive) {
            6 * BYTES_PER_GB
        } else {
            BYTES_PER_GB / 2
        };
        let headroom = self.prefill.max(margin);
        let bytes = self
            .device
            .saturating_sub(self.host)
            .saturating_sub(self.fixed)
            .saturating_sub(headroom)
            .min(available);

        ensure!(
            bytes > 0,
            "fixed buffers and reservations leave no expert pool capacity"
        );

        Ok(bytes)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/qwen4_exp/gpu/budget.rs"]
mod tests;
