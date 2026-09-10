//! Causal convolution and recurrent DeltaNet decode dispatch.

use super::*;

impl Gpu<'_> {
    /// Gated DeltaNet over `nb` rows; snapshots the state after rows
    /// 0..snap_after-1 for rollback. Input prepped in scratch.hc.h1.
    pub(super) fn deltanet_b(&self, enc: &Enc, d: &Delta, nb: usize, snap_after: usize) {
        let c = &self.p.cfg;
        let s = &self.scratch;
        let conv_dim = d.qkv.out;
        let nbu = nb as u32;
        let snap = snap_after as u32;
        self.qmv_h(enc, &d.qkv, &s.qkv, nb, &s.hc.h1);
        self.qmv_h(enc, &d.z, &s.z, nb, &s.hc.h1);
        self.qmv_h(enc, &d.a, &s.a, nb, &s.hc.h1);
        self.qmv_h(enc, &d.b, &s.b, nb, &s.hc.h1);
        let cp = ConvParams {
            channels: conv_dim,
            ksize: c.linear_conv_kernel_dim as u32,
        };
        self.dispatch(
            enc,
            &self.pipes.conv_b,
            |e| {
                self.bind(e, 0, &s.qkv, 0);
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
            snap_after: snap,
        };
        self.dispatch(
            enc,
            &self.pipes.delta_norms,
            |e| {
                self.bind(e, 0, &s.qkv, 0);
                self.bind(e, 1, &s.kqn, 0);
                set_bytes(e, 2, &p);
            },
            (nb * c.linear_num_key_heads).div_ceil(4),
            128,
            true,
        );
        self.dispatch(
            enc,
            &self.pipes.delta_gates,
            |e| {
                self.bind(e, 0, &s.a, 0);
                self.bind(e, 1, &s.b, 0);
                self.bind(e, 2, &self.dense, d.a_log.0);
                self.bind(e, 3, &self.dense, d.dt_bias.0);
                self.bind(e, 4, &s.gbuf, 0);
                set_bytes(e, 5, &p);
            },
            nb * c.linear_num_value_heads,
            96,
            false,
        );
        let rows = c.linear_num_value_heads * c.linear_value_head_dim;
        self.dispatch(
            enc,
            &self.pipes.delta_scan2,
            |e| {
                self.bind(e, 0, &s.qkv, 0);
                self.bind(e, 1, &s.kqn, 0);
                self.bind(e, 2, &s.gbuf, 0);
                self.bind(e, 3, &d.state, 0);
                self.bind(e, 4, &s.delta_y, 0);
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
                self.bind(e, 0, &s.delta_y, 0);
                self.bind(e, 1, &s.z, 0);
                self.bind(e, 2, &self.dense, d.norm.0);
                set_bytes(e, 3, &p);
            },
            nb * c.linear_num_value_heads,
            c.linear_value_head_dim,
            true,
        );
        self.prep_h(enc, &s.delta_y, 0, rows as u32, nb, &s.hc.h1);
        self.qmv_h(enc, &d.o, &s.mix_out, nb, &s.hc.h1);
    }
}
