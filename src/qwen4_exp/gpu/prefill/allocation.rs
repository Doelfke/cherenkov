//! Prefill scratch allocation and chunk capacity.

use super::*;
use crate::metal::PAGE_SIZE;
use std::cell::Cell;

// Both sizing and allocation walk this layout, including padding and the
// attention buffers that grow with context capacity.
fn scratch<T>(
    c: &crate::qwen4_exp::Qwen4ExpConfig,
    max_t: usize,
    rows: usize,
    all_logits: bool,
    allocate: impl Fn(usize) -> Result<T>,
) -> Result<PrefillScratch<T>> {
    let h = c.hidden_size;
    let hh = c.hc_hidden();
    let kv_row = c.num_key_value_heads * c.head_dim;
    let conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim
        + c.linear_num_value_heads * c.linear_value_head_dim;
    let v_dim = c.linear_num_value_heads * c.linear_value_head_dim;
    let inter = c.moe_intermediate_size;
    let k = c.num_experts_per_tok;
    let rp = rows.div_ceil(32) * 32 + 32;
    let max_blocks = max_t / c.indexer_compress_ratio + 1;
    let kl_pad = max_t.div_ceil(32) * 32;
    let n_rep = c.num_attention_heads / c.num_key_value_heads;
    let f = |n: usize| allocate(n * 4);
    let hf = |n: usize| allocate(n * 2);

    Ok(PrefillScratch {
        allocated_bytes: 0,
        rows: rp,
        ids: allocate(rp * 4)?,
        e: f(rp * h)?,
        hyper: f(rp * hh)?,
        normed: f(rp * hh)?,
        d: f(rp * c.hc_lowrank)?,
        u: f(rp * hh)?,
        mixed: f(rp * h)?,
        inj: f(rp * c.hc_count)?,
        mix_out: f(rp * h)?,
        moe_out: f(rp * h)?,
        qg: f(rp * c.num_attention_heads * 2 * c.head_dim)?,
        k: f(rp * kv_row)?,
        v: f(rp * kv_row)?,
        attn_out: f(rp * c.num_attention_heads * c.head_dim)?,
        iqk: f(rp * (c.indexer_n_heads + 1) * c.indexer_head_dim)?,
        iq: f(rp * c.indexer_n_heads * c.indexer_head_dim)?,
        bscore: f(rp * max_blocks)?,
        vis: allocate(rp * (c.indexer_budget + c.indexer_compress_ratio) * 4)?,
        nvis: allocate(rp * 4)?,
        vmask: allocate(rp * max_blocks.div_ceil(32) * 4)?,
        ag_qh: hf(c.num_attention_heads * QS * c.head_dim)?,
        ag_kh: hf(kl_pad * c.head_dim)?,
        ag_vt: hf(c.head_dim * kl_pad)?,
        ag_s: f(n_rep * QS * kl_pad)?,
        ag_p: hf(n_rep * QS * kl_pad)?,
        ag_o: f(n_rep * QS * c.head_dim)?,
        qkv: f(rp * conv_dim)?,
        z: f(rp * v_dim)?,
        a: f(rp * c.linear_num_value_heads)?,
        b: f(rp * c.linear_num_value_heads)?,
        kqn: f(rp * 2 * c.linear_num_key_heads * c.linear_key_head_dim)?,
        gbuf: f(rp * c.linear_num_value_heads * 2)?,
        delta_y: f(rp * v_dim)?,
        router: f(rp * c.num_experts)?,
        topk_idx: allocate(rp * k * 4)?,
        topk_w: f(rp * k)?,
        csr_rows: allocate(rp * k * 4)?,
        csr_w: f(rp * k)?,
        xg: f(rp * h)?,
        ge: f(rp * inter)?,
        ue: f(rp * inter)?,
        hg: f(rp * inter)?,
        ye: f(rp * h)?,
        mtp_hyper: f(rp * hh)?,
        fe: f(rp * h)?,
        fh: f(rp * hh)?,
        logits_all: if all_logits {
            Some(f(rp * c.vocab_size)?)
        } else {
            None
        },
    })
}

pub(in crate::qwen4_exp::gpu) fn scratch_bytes(
    c: &crate::qwen4_exp::Qwen4ExpConfig,
    max_t: usize,
    rows: usize,
    all_logits: bool,
) -> Result<usize> {
    let total = Cell::new(0usize);

    scratch(c, max_t, rows, all_logits, |bytes| {
        // Metal may round each resource to a page. Budget that rounding too.
        let padded = bytes.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let sum = total
            .get()
            .checked_add(padded)
            .context("prefill scratch size overflow")?;

        total.set(sum);

        Ok(())
    })?;

    Ok(total.get())
}

fn rows_fit(
    c: &crate::qwen4_exp::Qwen4ExpConfig,
    max_t: usize,
    available: usize,
    all_logits: bool,
) -> Result<usize> {
    let (mut low, mut high) = (0, MAX_PREFILL_ROWS.min(max_t));

    while low < high {
        let mid = low + (high - low).div_ceil(2);

        if scratch_bytes(c, max_t, mid, all_logits)? <= available {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    ensure!(
        low > 0,
        "insufficient memory for prefill scratch; reduce expert pool or context capacity"
    );

    Ok(low)
}

impl Gpu<'_> {
    pub(super) fn pf_alloc(&self, rows: usize, all_logits: bool) -> Result<PrefillScratch> {
        let before = self.allocated_bytes();
        let mut buffers = scratch(&self.p.cfg, self.max_t, rows, all_logits, |bytes| {
            self.ctx.new_buffer(bytes)
        })?;
        buffers.allocated_bytes = self.allocated_bytes().saturating_sub(before) as usize;

        Ok(buffers)
    }

    /// Maximum chunk that fits the configured budget, or the device recommendation.
    pub fn prefill_rows_fit(&self, all_logits: bool) -> Result<usize> {
        use objc2_metal::MTLDevice as _;

        let device_limit = self.ctx.device.recommendedMaxWorkingSetSize() as usize;
        let limit = self.ctx.allocation_limit.get().unwrap_or(device_limit);
        let mut used = self.ctx.device.currentAllocatedSize();

        // Existing scratch is reused or released before its replacement.
        if let Some(pf) = &self.pf {
            used = used.saturating_sub(pf.allocated_bytes);
        }

        rows_fit(
            &self.p.cfg,
            self.max_t,
            limit.saturating_sub(used),
            all_logits,
        )
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/qwen4_exp/gpu/prefill_allocation.rs"]
mod tests;
