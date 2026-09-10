//! Prefill DeltaNet convolution, scan, and output gate.

use super::*;

impl Gpu<'_> {
    /// Gated DeltaNet over t rows (sequential scan). pf.mixed -> pf.mix_out.
    pub(super) fn pf_deltanet(&self, enc: &Enc, d: &Delta, t: usize, pf: &PrefillScratch) {
        let c = &self.p.cfg;
        let conv_dim = d.qkv.out;
        let nbu = t as u32;

        self.qmm(enc, &d.qkv, &pf.mixed, &pf.qkv, t);
        self.qmm(enc, &d.z, &pf.mixed, &pf.z, t);
        self.qmm(enc, &d.a, &pf.mixed, &pf.a, t);
        self.qmm(enc, &d.b, &pf.mixed, &pf.b, t);

        let cp = ConvParams {
            channels: conv_dim,
            ksize: c.linear_conv_kernel_dim as u32,
        };
        let snap = 0u32;

        self.dispatch(
            enc,
            &self.pipes.conv_b,
            |e| {
                self.bind(e, 0, &pf.qkv, 0);
                self.bind(e, 1, &self.dense, d.conv.0);
                self.bind(e, 2, &d.hist, 0);
                set_bytes(e, 3, &cp);
                set_bytes(e, 4, &nbu);
                set_bytes(e, 5, &snap);
                self.bind(e, 6, &d.mid_hist, 0);
            },
            conv_dim as usize,
            256,
            false,
        );

        let p = DeltaPrepParams {
            n_k: c.linear_num_key_heads as u32,
            n_v: c.linear_num_value_heads as u32,
            d_k: c.linear_key_head_dim as u32,
            d_v: c.linear_value_head_dim as u32,
            eps: 1e-6,
            nb: nbu,
            snap_after: 0,
        };

        self.dispatch(
            enc,
            &self.pipes.delta_norms,
            |e| {
                self.bind(e, 0, &pf.qkv, 0);
                self.bind(e, 1, &pf.kqn, 0);
                set_bytes(e, 2, &p);
            },
            (t * c.linear_num_key_heads).div_ceil(4),
            128,
            true,
        );
        self.dispatch(
            enc,
            &self.pipes.delta_gates,
            |e| {
                self.bind(e, 0, &pf.a, 0);
                self.bind(e, 1, &pf.b, 0);
                self.bind(e, 2, &self.dense, d.a_log.0);
                self.bind(e, 3, &self.dense, d.dt_bias.0);
                self.bind(e, 4, &pf.gbuf, 0);
                set_bytes(e, 5, &p);
            },
            t * c.linear_num_value_heads,
            96,
            false,
        );

        let rows = c.linear_num_value_heads * c.linear_value_head_dim;

        self.dispatch(
            enc,
            &self.pipes.delta_scan2,
            |e| {
                self.bind(e, 0, &pf.qkv, 0);
                self.bind(e, 1, &pf.kqn, 0);
                self.bind(e, 2, &pf.gbuf, 0);
                self.bind(e, 3, &d.state, 0);
                self.bind(e, 4, &pf.delta_y, 0);
                self.bind(e, 5, &d.mid, 0);
                set_bytes(e, 6, &p);
            },
            rows.div_ceil(4),
            128,
            true,
        );
        self.dispatch(
            enc,
            &self.pipes.gate_norm_sigmoid_b,
            |e| {
                self.bind(e, 0, &pf.delta_y, 0);
                self.bind(e, 1, &pf.z, 0);
                self.bind(e, 2, &self.dense, d.norm.0);
                set_bytes(e, 3, &p);
            },
            t * c.linear_num_value_heads,
            c.linear_value_head_dim,
            true,
        );
        self.qmm(enc, &d.o, &pf.delta_y, &pf.mix_out, t);
    }
}
