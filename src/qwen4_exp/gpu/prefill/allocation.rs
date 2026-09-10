//! Prefill scratch allocation and chunk capacity.

use super::*;

impl Gpu<'_> {
    pub(super) fn pf_alloc(&self, rows: usize, all_logits: bool) -> Result<PrefillScratch> {
        let c = &self.p.cfg;
        let ctx = &self.ctx;
        let h = c.hidden_size;
        let hh = c.hc_hidden();
        let kv_row = c.num_key_value_heads * c.head_dim;
        let conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim
            + c.linear_num_value_heads * c.linear_value_head_dim;
        let v_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        let inter = c.moe_intermediate_size;
        let k = c.num_experts_per_tok;
        let rp = rows.div_ceil(32) * 32 + 32;
        let max_blocks = self.max_t / c.indexer_compress_ratio + 1;
        let kl_pad = self.max_t.div_ceil(32) * 32;
        let n_rep = c.num_attention_heads / c.num_key_value_heads;
        let f = |n: usize| ctx.new_buffer(n * 4);
        let hf = |n: usize| ctx.new_buffer(n * 2);
        Ok(PrefillScratch {
            rows: rp,
            ids: ctx.new_buffer(rp * 4)?,
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
            vis: ctx.new_buffer(rp * (c.indexer_budget + c.indexer_compress_ratio) * 4)?,
            nvis: ctx.new_buffer(rp * 4)?,
            vmask: ctx.new_buffer(rp * max_blocks.div_ceil(32) * 4)?,
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
            topk_idx: ctx.new_buffer(rp * k * 4)?,
            topk_w: f(rp * k)?,
            csr_rows: ctx.new_buffer(rp * k * 4)?,
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

    /// Rows per prefill chunk that fit the Metal headroom (1 GB kept
    /// free), between 512 and 4,096.
    pub fn prefill_rows_fit(&self) -> usize {
        use objc2_metal::MTLDevice as _;
        let device_limit = self.ctx.device.recommendedMaxWorkingSetSize() as usize;
        let limit = self
            .ctx
            .allocation_limit
            .get()
            .unwrap_or(device_limit)
            .min(device_limit) as f64;
        let used = self.ctx.device.currentAllocatedSize() as f64;
        let rows = ((limit - used - 1e9) / ROW_BYTES as f64).max(0.0) as usize;
        (rows / 256 * 256).clamp(512, 4096)
    }
}
