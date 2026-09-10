//! QSA indexing, q8 KV append, and selected decode attention.

use super::*;

impl Gpu<'_> {
    /// Full attention over `nb` rows at positions base_pos..: dense while
    /// a row's context fits the indexer budget, otherwise over the blocks
    /// the QSA indexer selects for that row. Input prepped in
    /// scratch.hc.h1; output in scratch.mix_out.
    pub(super) fn attention_b(&self, enc: &Enc, a: &Attn, base_pos: usize, nb: usize) {
        let c = &self.p.cfg;
        let s = &self.scratch;
        let hd = c.head_dim as u32;
        let kv_row = c.num_key_value_heads * c.head_dim;
        let nbu = nb as u32;
        self.qmv_h(enc, &a.q, &s.qg, nb, &s.hc.h1);
        self.qmv_h(enc, &a.k, &s.k, nb, &s.hc.h1);
        self.qmv_h(enc, &a.v, &s.v, nb, &s.hc.h1);
        self.qmv_h(enc, &a.iqk, &s.iqk, nb, &s.hc.h1);
        let rot = (c.head_dim as f64 * c.partial_rotary_factor) as u32;
        // Indexer: cache this batch's raw keys, refresh the blocks it
        // completes, and pick blocks for rows past the budget.
        let ratio = c.indexer_compress_ratio;
        let ihd = c.indexer_head_dim;
        let inh = c.indexer_n_heads;
        let kblk = c.indexer_budget / ratio;
        let max_blocks = self.max_t / ratio + 1;
        let vis_stride = c.indexer_budget + ratio;
        let ip = IndexParams {
            ihd: ihd as u32,
            inh: inh as u32,
            qk_dim: a.iqk.out,
            ratio: ratio as u32,
            rot,
            theta: c.rope_parameters.rope_theta as f32,
            eps: c.rms_norm_eps as f32,
            base_pos: base_pos as u32,
            nb: nbu,
            b0: (base_pos / ratio) as u32,
            b1: ((base_pos + nb) / ratio) as u32,
            k: kblk as u32,
            max_blocks: max_blocks as u32,
            vis_stride: vis_stride as u32,
            mask_words: max_blocks.div_ceil(32) as u32,
        };
        self.dispatch(
            enc,
            &self.pipes.index_append,
            |e| {
                self.bind(e, 0, &s.iqk, 0);
                self.bind(e, 1, &a.ikc, 0);
                set_bytes(e, 2, &ip);
            },
            nb * ihd,
            256,
            false,
        );
        if ip.b1 > ip.b0 {
            self.dispatch(
                enc,
                &self.pipes.index_blocks,
                |e| {
                    self.bind(e, 0, &a.ikc, 0);
                    self.bind(e, 1, &a.blk, 0);
                    self.bind(e, 2, &self.dense, a.ikn.0);
                    set_bytes(e, 3, &ip);
                },
                (ip.b1 - ip.b0) as usize,
                ihd,
                true,
            );
        }
        let any_selective = (base_pos + nb) / ratio > kblk;
        if any_selective {
            self.dispatch(
                enc,
                &self.pipes.index_q,
                |e| {
                    self.bind(e, 0, &s.iqk, 0);
                    self.bind(e, 1, &s.iq, 0);
                    self.bind(e, 2, &self.dense, a.iqn.0);
                    set_bytes(e, 3, &ip);
                },
                nb * inh,
                ihd,
                true,
            );
            self.dispatch(
                enc,
                &self.pipes.index_score,
                |e| {
                    self.bind(e, 0, &s.iq, 0);
                    self.bind(e, 1, &a.blk, 0);
                    self.bind(e, 2, &s.bscore, 0);
                    set_bytes(e, 3, &ip);
                },
                nb * max_blocks.div_ceil(256),
                256,
                true,
            );
            self.dispatch(
                enc,
                &self.pipes.index_select,
                |e| {
                    self.bind(e, 0, &s.bscore, 0);
                    self.bind(e, 1, &s.vis, 0);
                    self.bind(e, 2, &s.nvis, 0);
                    set_bytes(e, 3, &ip);
                    self.bind(e, 4, &s.vmask, 0);
                },
                nb,
                1024,
                true,
            );
        }
        let qp = QkRopeParams {
            n_heads: c.num_attention_heads as u32,
            head_dim: hd,
            stride: 2 * hd,
            rot,
            pos: base_pos as u32,
            theta: c.rope_parameters.rope_theta as f32,
            eps: c.rms_norm_eps as f32,
        };
        self.dispatch(
            enc,
            &self.pipes.qk_norm_rope_b,
            |e| {
                self.bind(e, 0, &s.qg, 0);
                self.bind(e, 1, &self.dense, a.qn.0);
                set_bytes(e, 2, &qp);
                set_bytes(e, 3, &nbu);
            },
            (nb * c.num_attention_heads).div_ceil(4),
            128,
            true,
        );
        let kp = QkRopeParams {
            n_heads: c.num_key_value_heads as u32,
            stride: hd,
            ..qp
        };
        self.dispatch(
            enc,
            &self.pipes.qk_norm_rope_b,
            |e| {
                self.bind(e, 0, &s.k, 0);
                self.bind(e, 1, &self.dense, a.kn.0);
                set_bytes(e, 2, &kp);
                set_bytes(e, 3, &nbu);
            },
            (nb * c.num_key_value_heads).div_ceil(4),
            128,
            true,
        );
        let kq = KvQParams {
            row: kv_row as u32,
            t0: base_pos as u32,
            nb: nbu,
            scale_off: kv_q8_side(self.max_t, kv_row).0 as u32,
        };
        let sgs = nb * (kv_row / 32) * 2;
        self.dispatch(
            enc,
            &self.pipes.kv_append_q8,
            |e| {
                self.bind(e, 0, &s.k, 0);
                self.bind(e, 1, &s.v, 0);
                self.bind(e, 2, &a.kc, 0);
                self.bind(e, 3, &a.vc, 0);
                set_bytes(e, 4, &kq);
            },
            sgs.div_ceil(4),
            128,
            true,
        );
        // Split-T flash decode per row over its causal prefix, or over the
        // indexer's visible token list once the row is past the budget.
        let n_rep = c.num_attention_heads / c.num_key_value_heads;
        for b in 0..nb {
            let t_len = base_pos + b + 1;
            let blocks = t_len / ratio;
            let selective = blocks > kblk;
            let n_vis = if selective {
                kblk * ratio + (t_len - blocks * ratio)
            } else {
                t_len
            };
            let n_chunks = n_vis.div_ceil(32);
            let n_wg = n_chunks.div_ceil(3).clamp(1, ATTN_MAX_WG);
            let ap = AttnPartParams {
                n_heads: c.num_attention_heads as u32,
                n_kv: c.num_key_value_heads as u32,
                head_dim: hd,
                t_len: n_vis as u32,
                q_stride: 2 * hd,
                q_off: (b * c.num_attention_heads * 2 * c.head_dim) as u32,
                max_blk: self.max_t.div_ceil(ATTN_TB).max(ATTN_MAX_WG) as u32,
                scale: (c.head_dim as f32).powf(-0.5) * std::f32::consts::LOG2_E,
            };
            let n_wg_u = n_wg as u32;
            let scale_off = kq.scale_off;
            let pipe = if selective {
                &self.pipes.attn_sel
            } else {
                &self.pipes.attn_part2_q8
            };
            self.dispatch(
                enc,
                pipe,
                |e| {
                    self.bind(e, 0, &s.qg, 0);
                    self.bind(e, 1, &a.kc, 0);
                    self.bind(e, 2, &a.vc, 0);
                    self.bind(e, 3, &s.attn_parts, 0);
                    set_bytes(e, 4, &ap);
                    set_bytes(e, 5, &n_wg_u);
                    set_bytes(e, 6, &scale_off);
                    if selective {
                        self.bind(e, 7, &s.vis, b * vis_stride * 4);
                    }
                },
                c.num_key_value_heads * n_wg,
                32 * n_rep,
                true,
            );
            let nblk = n_wg as u32;
            let out_off = (b * c.num_attention_heads * c.head_dim) as u32;
            self.dispatch(
                enc,
                &self.pipes.attn_combine,
                |e| {
                    self.bind(e, 0, &s.qg, 0);
                    self.bind(e, 1, &s.attn_parts, 0);
                    self.bind(e, 2, &s.attn_out, 0);
                    set_bytes(e, 3, &ap);
                    set_bytes(e, 4, &nblk);
                    set_bytes(e, 5, &out_off);
                },
                c.num_attention_heads,
                256,
                true,
            );
        }
        self.prep_h(
            enc,
            &s.attn_out,
            0,
            (c.num_attention_heads * c.head_dim) as u32,
            nb,
            &s.hc.h1,
        );
        self.qmv_h(enc, &a.o, &s.mix_out, nb, &s.hc.h1);
    }
}
