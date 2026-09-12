//! Runtime resource counters shared by CLI telemetry and control status.

use super::*;
use objc2_metal::MTLDevice;
use serde::Serialize;

/// Capacities and observations, not a partition of process memory. CPU prefix
/// checkpoints and session history are reported separately by the server.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct MemoryStats {
    pub metal_allocated_bytes_observed: u64,
    /// Length of the shared weight buffer, counted once for all tensor views.
    pub mapped_weight_buffer_bytes: usize,
    pub kv_index_capacity_bytes: usize,
    pub context_capacity_tokens: usize,
    pub mtp_enabled: bool,
    /// Configured pool capacity; the residency-set backend may use fewer slots.
    pub expert_pool_bytes: usize,
    pub device_working_set_bytes: usize,
    pub allocation_limit_bytes: Option<usize>,
    pub prefill_reserved_bytes: usize,
    pub expert_pool_slots: usize,
    pub resident_experts: usize,
}

impl Gpu<'_> {
    pub fn allocated_bytes(&self) -> u64 {
        self.ctx.device.currentAllocatedSize() as u64
    }

    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            metal_allocated_bytes_observed: self.allocated_bytes(),
            mapped_weight_buffer_bytes: self.dense.length(),
            kv_index_capacity_bytes: self.kv_index_capacity_bytes(),
            context_capacity_tokens: self.max_t,
            mtp_enabled: self.has_mtp(),
            expert_pool_bytes: self.pool_bytes(),
            device_working_set_bytes: self.ctx.device.recommendedMaxWorkingSetSize() as usize,
            allocation_limit_bytes: self.ctx.allocation_limit.get(),
            prefill_reserved_bytes: self.prefill_reserved_bytes,
            expert_pool_slots: self.pool_slots(),
            resident_experts: self.pool_resident(),
        }
    }

    /// Reserved KV and attention-index bytes, including the enabled MTP layer.
    fn kv_index_capacity_bytes(&self) -> usize {
        self.layers
            .iter()
            .chain(self.mtp.iter().map(|m| &m.layer))
            .map(|layer| match &layer.mix {
                Mix::Attn(a) => a.kc.length() + a.vc.length() + a.ikc.length() + a.blk.length(),
                Mix::Delta(_) => 0,
            })
            .sum()
    }
}
