//! Grouped normalization and gated residual mixing/injection.

use super::*;

impl Gpu<'_> {
    pub(super) fn group_params(&self, eps: bool, shift: f32) -> GroupParams {
        let c = &self.p.cfg;

        GroupParams {
            n: c.hidden_size as u32,
            groups: c.hc_count as u32,
            eps: if eps { c.rms_norm_eps as f32 } else { 0.0 },
            shift,
        }
    }

    /// Grouped RMSNorm of `nb` rows (each `groups` slices of `n`), weight
    /// shifted by `shift` (1.0 for raw HF norm weights).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn group_norm_b(
        &self,
        enc: &Enc,
        x: &Buf,
        x_off: usize,
        w: T,
        y: &Buf,
        n: u32,
        groups: u32,
        shift: f32,
        nb: usize,
    ) {
        let p = GroupParams {
            n,
            groups,
            eps: self.p.cfg.rms_norm_eps as f32,
            shift,
        };

        self.dispatch(
            enc,
            &self.pipes.group_norm_b,
            |e| {
                self.bind(e, 0, x, x_off);
                self.bind(e, 1, &self.dense, w.0);
                self.bind(e, 2, y, 0);
                set_bytes(e, 3, &p);
            },
            nb * groups as usize,
            256,
            true,
        );
    }

    /// Gated residual read of `nb` rows of `hyper` for one block: the block
    /// input lands in `bufs.mixed` and, prepped, in `bufs.h1`; injection
    /// logits in `bufs.inj`. `pending` is a block output not yet injected
    /// into the stream; the fused norm applies it first (in place).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_read_b(
        &self,
        enc: &Enc,
        hc: &Hc,
        nb: usize,
        hyper: &Buf,
        hyper_off: usize,
        pending: Option<&Buf>,
        bufs: &HcBufs,
        inject: bool,
    ) {
        let c = &self.p.cfg;
        let h = c.hidden_size as u32;
        let nbu = nb as u32;
        let np = NormPrepParams {
            n: h,
            groups: c.hc_count as u32,
            eps: c.rms_norm_eps as f32,
            inject: pending.is_some() as u32,
        };
        let out = pending.unwrap_or(&self.scratch.mix_out);

        self.dispatch(
            enc,
            &self.pipes.norm_prep_b,
            |e| {
                self.bind(e, 0, hyper, hyper_off);
                self.bind(e, 1, &self.dense, hc.norm.0);
                self.bind(e, 2, &bufs.normed, 0);
                self.bind(e, 3, &bufs.h1.xe, 0);
                self.bind(e, 4, &bufs.h1.xo, 0);
                self.bind(e, 5, &bufs.h1.xsum, 0);
                self.bind(e, 6, out, 0);
                self.bind(e, 7, &bufs.inj, 0);
                set_bytes(e, 8, &np);
            },
            nb * c.hc_count,
            256,
            true,
        );

        // Bottleneck down-projection with silu(./hc).
        let div = c.hc_count as f32;
        let qp = QmvParams {
            out_dim: hc.down.out,
            in_dim: hc.down.inp,
        };

        self.dispatch(
            enc,
            &self.pipes.qmv_silu_b[nb - 1],
            |e| {
                self.bind(e, 0, &self.dense, hc.down.w);
                self.bind(e, 1, &self.dense, hc.down.s);
                self.bind(e, 2, &self.dense, hc.down.b);
                self.bind(e, 3, &bufs.h1.xe, 0);
                self.bind(e, 4, &bufs.h1.xo, 0);
                self.bind(e, 5, &bufs.h1.xsum, 0);
                self.bind(e, 6, &bufs.d, 0);
                set_bytes(e, 7, &qp);
                set_bytes(e, 8, &div);
            },
            (hc.down.out as usize).div_ceil(4),
            128,
            true,
        );

        // Injection logits from the normed stream.
        if inject && let Some(inj) = &hc.inject {
            self.qmv_h(enc, inj, &bufs.inj, nb, &bufs.h1);
        }

        // Up-projection back to the stream width, then the mix.
        self.prep_h(enc, &bufs.d, 0, hc.down.out, nb, &bufs.h2);
        self.qmv_h(enc, &hc.up, &bufs.u, nb, &bufs.h2);

        let gp = self.group_params(false, 0.0);

        self.dispatch(
            enc,
            &self.pipes.hc_mix_b,
            |e| {
                self.bind(e, 0, &bufs.u, 0);
                self.bind(e, 1, &bufs.normed, 0);
                self.bind(e, 2, &bufs.mixed, 0);
                set_bytes(e, 3, &gp);
                set_bytes(e, 4, &nbu);
            },
            nb * h as usize,
            256,
            false,
        );
        self.prep_h(enc, &bufs.mixed, 0, h, nb, &bufs.h1);
    }

    pub(super) fn inject_b(&self, enc: &Enc, hyper: &Buf, out: &Buf, nb: usize) {
        let c = &self.p.cfg;
        let gp = self.group_params(false, 0.0);
        let nbu = nb as u32;

        self.dispatch(
            enc,
            &self.pipes.inject_b,
            |e| {
                self.bind(e, 0, hyper, 0);
                self.bind(e, 1, out, 0);
                self.bind(e, 2, &self.scratch.hc.inj, 0);
                set_bytes(e, 3, &gp);
                set_bytes(e, 4, &nbu);
            },
            nb * c.hc_hidden(),
            256,
            false,
        );
    }
}
