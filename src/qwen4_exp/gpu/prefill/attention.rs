//! QSA masks and GEMM attention over query sub-chunks.

use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnGemmParams {
    kv_row: u32,
    blocks_per_row: u32,
    head: u32,
    kl: u32,
    kl_pad: u32,
    scale_off: u32,
    base: u32,
    nb: u32,
    n_rep: u32,
    scale: f32,
}

#[repr(C)]
struct GemmHParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
}

#[repr(C)]
struct SelMaskParams {
    row0: u32,
    mask_words: u32,
    ratio: u32,
    k: u32,
}

impl Gpu<'_> {
    /// Full attention over t rows at positions base..: GEMM form in
    /// sub-chunks of QS queries, with the QSA mask for rows past the
    /// budget. Input pf.mixed, output pf.mix_out.
    pub(super) fn pf_attention(
        &self,
        enc: &Enc,
        a: &Attn,
        base: usize,
        t: usize,
        pf: &PrefillScratch,
    ) {
        let c = &self.p.cfg;
        let hd = c.head_dim;
        let n_heads = c.num_attention_heads;
        let n_kv = c.num_key_value_heads;
        let n_rep = n_heads / n_kv;
        let kv_row = n_kv * hd;
        let nbu = t as u32;

        self.qmm(enc, &a.q, &pf.mixed, &pf.qg, t);
        self.qmm(enc, &a.k, &pf.mixed, &pf.k, t);
        self.qmm(enc, &a.v, &pf.mixed, &pf.v, t);
        self.qmm(enc, &a.iqk, &pf.mixed, &pf.iqk, t);

        let rot = (hd as f64 * c.partial_rotary_factor) as u32;
        let qp = QkRopeParams {
            n_heads: n_heads as u32,
            head_dim: hd as u32,
            stride: 2 * hd as u32,
            rot,
            pos: base as u32,
            theta: c.rope_parameters.rope_theta as f32,
            eps: c.rms_norm_eps as f32,
        };
        self.dispatch(
            enc,
            &self.pipes.qk_norm_rope_b,
            |e| {
                self.bind(e, 0, &pf.qg, 0);
                self.bind(e, 1, &self.dense, a.qn.0);
                set_bytes(e, 2, &qp);
                set_bytes(e, 3, &nbu);
            },
            (t * n_heads).div_ceil(4),
            128,
            true,
        );

        let kp = QkRopeParams {
            n_heads: n_kv as u32,
            stride: hd as u32,
            ..qp
        };
        self.dispatch(
            enc,
            &self.pipes.qk_norm_rope_b,
            |e| {
                self.bind(e, 0, &pf.k, 0);
                self.bind(e, 1, &self.dense, a.kn.0);
                set_bytes(e, 2, &kp);
                set_bytes(e, 3, &nbu);
            },
            (t * n_kv).div_ceil(4),
            128,
            true,
        );

        // Indexer: raw keys, block keys, and block selection for rows past
        // the budget (as a bitmask for the softmax).
        let ratio = c.indexer_compress_ratio;
        let ihd = c.indexer_head_dim;
        let inh = c.indexer_n_heads;
        let kblk = c.indexer_budget / ratio;
        let max_blocks = self.max_t / ratio + 1;
        let mask_words = max_blocks.div_ceil(32);
        let ip = IndexParams {
            ihd: ihd as u32,
            inh: inh as u32,
            qk_dim: a.iqk.out,
            ratio: ratio as u32,
            rot,
            theta: c.rope_parameters.rope_theta as f32,
            eps: c.rms_norm_eps as f32,
            base_pos: base as u32,
            nb: nbu,
            b0: (base / ratio) as u32,
            b1: ((base + t) / ratio) as u32,
            k: kblk as u32,
            max_blocks: max_blocks as u32,
            vis_stride: (c.indexer_budget + ratio) as u32,
            mask_words: mask_words as u32,
        };
        self.dispatch(
            enc,
            &self.pipes.index_append,
            |e| {
                self.bind(e, 0, &pf.iqk, 0);
                self.bind(e, 1, &a.ikc, 0);
                set_bytes(e, 2, &ip);
            },
            t * ihd,
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

        if (base + t) / ratio > kblk {
            self.dispatch(
                enc,
                &self.pipes.index_q,
                |e| {
                    self.bind(e, 0, &pf.iqk, 0);
                    self.bind(e, 1, &pf.iq, 0);
                    self.bind(e, 2, &self.dense, a.iqn.0);
                    set_bytes(e, 3, &ip);
                },
                t * inh,
                ihd,
                true,
            );

            self.dispatch(
                enc,
                &self.pipes.index_score,
                |e| {
                    self.bind(e, 0, &pf.iq, 0);
                    self.bind(e, 1, &a.blk, 0);
                    self.bind(e, 2, &pf.bscore, 0);
                    set_bytes(e, 3, &ip);
                },
                t * max_blocks.div_ceil(256),
                256,
                true,
            );

            self.dispatch(
                enc,
                &self.pipes.index_select,
                |e| {
                    self.bind(e, 0, &pf.bscore, 0);
                    self.bind(e, 1, &pf.vis, 0);
                    self.bind(e, 2, &pf.nvis, 0);
                    set_bytes(e, 3, &ip);
                    self.bind(e, 4, &pf.vmask, 0);
                },
                t,
                1024,
                true,
            );
        }

        let scale_off = kv_q8_side(self.max_t, kv_row).0;
        let kq = KvQParams {
            row: kv_row as u32,
            t0: base as u32,
            nb: nbu,
            scale_off: scale_off as u32,
        };
        self.dispatch(
            enc,
            &self.pipes.kv_append_q8,
            |e| {
                self.bind(e, 0, &pf.k, 0);
                self.bind(e, 1, &pf.v, 0);
                self.bind(e, 2, &a.kc, 0);
                self.bind(e, 3, &a.vc, 0);
                set_bytes(e, 4, &kq);
            },
            (t * (kv_row / 32) * 2).div_ceil(4),
            128,
            true,
        );

        // GEMM attention per query sub-chunk and kv head.
        let n_heads_u = n_heads as u32;
        let mut q0 = 0;
        while q0 < t {
            let n = (t - q0).min(QS);
            let base_q = base + q0;
            let kl = base_q + n;
            let kl_pad = kl.div_ceil(32) * 32;
            let p = AttnGemmParams {
                kv_row: kv_row as u32,
                blocks_per_row: (kv_row / 32) as u32,
                head: 0,
                kl: kl as u32,
                kl_pad: kl_pad as u32,
                scale_off: scale_off as u32,
                base: base_q as u32,
                nb: n as u32,
                n_rep: n_rep as u32,
                scale: (hd as f32).powf(-0.5) * std::f32::consts::LOG2_E,
            };
            let qg_off = q0 * n_heads * 2 * hd * 4;
            self.dispatch(
                enc,
                &self.pipes.attn_q_stage,
                |e| {
                    self.bind(e, 0, &pf.qg, qg_off);
                    self.bind(e, 1, &pf.ag_qh, 0);
                    set_bytes(e, 2, &p);
                    set_bytes(e, 3, &n_heads_u);
                },
                n_heads * n * hd,
                256,
                false,
            );

            let m = n_rep * n;
            let sm = SelMaskParams {
                row0: q0 as u32,
                mask_words: mask_words as u32,
                ratio: ratio as u32,
                k: kblk as u32,
            };
            for hk in 0..n_kv {
                let ph = AttnGemmParams {
                    head: hk as u32,
                    ..p
                };
                self.dispatch(
                    enc,
                    &self.pipes.attn_kv_stage,
                    |e| {
                        self.bind(e, 0, &a.kc, 0);
                        self.bind(e, 1, &a.vc, 0);
                        self.bind(e, 2, &pf.ag_kh, 0);
                        self.bind(e, 3, &pf.ag_vt, 0);
                        set_bytes(e, 4, &ph);
                    },
                    kl_pad * hd,
                    256,
                    false,
                );

                let ps = GemmHParams {
                    m: m as u32,
                    n: kl_pad as u32,
                    k: hd as u32,
                    lda: hd as u32,
                    ldb: hd as u32,
                    ldc: kl_pad as u32,
                };
                self.dispatch(
                    enc,
                    &self.pipes.gemm_hh,
                    |e| {
                        self.bind(e, 0, &pf.ag_qh, hk * n_rep * n * hd * 2);
                        self.bind(e, 1, &pf.ag_kh, 0);
                        self.bind(e, 2, &pf.ag_s, 0);
                        set_bytes(e, 3, &ps);
                    },
                    m.div_ceil(128) * (kl_pad / 32),
                    128,
                    true,
                );

                self.dispatch(
                    enc,
                    &self.pipes.softmax_sel,
                    |e| {
                        self.bind(e, 0, &pf.ag_s, 0);
                        self.bind(e, 1, &pf.ag_p, 0);
                        set_bytes(e, 2, &ph);
                        self.bind(e, 3, &pf.vmask, 0);
                        set_bytes(e, 4, &sm);
                    },
                    m,
                    256,
                    true,
                );

                let po = GemmHParams {
                    m: m as u32,
                    n: hd as u32,
                    k: kl_pad as u32,
                    lda: kl_pad as u32,
                    ldb: kl_pad as u32,
                    ldc: hd as u32,
                };
                self.dispatch(
                    enc,
                    &self.pipes.gemm_hh,
                    |e| {
                        self.bind(e, 0, &pf.ag_p, 0);
                        self.bind(e, 1, &pf.ag_vt, 0);
                        self.bind(e, 2, &pf.ag_o, 0);
                        set_bytes(e, 3, &po);
                    },
                    m.div_ceil(128) * (hd / 32),
                    128,
                    true,
                );

                self.dispatch(
                    enc,
                    &self.pipes.attn_o_scatter,
                    |e| {
                        self.bind(e, 0, &pf.ag_o, 0);
                        self.bind(e, 1, &pf.qg, qg_off);
                        self.bind(e, 2, &pf.attn_out, q0 * n_heads * hd * 4);
                        set_bytes(e, 3, &ph);
                        set_bytes(e, 4, &n_heads_u);
                    },
                    n_rep * n * hd,
                    256,
                    false,
                );
            }

            q0 += n;
        }

        self.qmm(enc, &a.o, &pf.attn_out, &pf.mix_out, t);
    }
}
