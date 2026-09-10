//! Prefill grouped residual reads and injection.

use super::*;

impl Gpu<'_> {
    pub(super) fn pf_inject(&self, enc: &Enc, hyper: &Buf, out: &Buf, inj: &Buf, t: usize) {
        let gp = self.group_params(false, 0.0);
        let nbu = t as u32;

        self.dispatch(
            enc,
            &self.pipes.inject_b,
            |e| {
                self.bind(e, 0, hyper, 0);
                self.bind(e, 1, out, 0);
                self.bind(e, 2, inj, 0);
                set_bytes(e, 3, &gp);
                set_bytes(e, 4, &nbu);
            },
            t * self.p.cfg.hc_hidden(),
            256,
            false,
        );
    }

    /// Gated residual read over t rows: pf.mixed (block input), pf.inj.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pf_hc_read(
        &self,
        enc: &Enc,
        hc: &Hc,
        t: usize,
        pf: &PrefillScratch,
        hyper: &Buf,
        pending: Option<&Buf>,
        inject: bool,
    ) {
        let c = &self.p.cfg;
        let h = c.hidden_size as u32;

        if let Some(out) = pending {
            self.pf_inject(enc, hyper, out, &pf.inj, t);
        }

        self.group_norm_b(
            enc,
            hyper,
            0,
            hc.norm,
            &pf.normed,
            h,
            c.hc_count as u32,
            0.0,
            t,
        );
        self.qmm(enc, &hc.down, &pf.normed, &pf.d, t);

        let n = (t as u32) * hc.down.out;
        let div = c.hc_count as f32;

        self.dispatch(
            enc,
            &self.pipes.silu_rows,
            |e| {
                self.bind(e, 0, &pf.d, 0);
                set_bytes(e, 1, &n);
                set_bytes(e, 2, &div);
            },
            n as usize,
            256,
            false,
        );
        self.qmm(enc, &hc.up, &pf.d, &pf.u, t);

        if inject && let Some(q) = &hc.inject {
            self.qmm(enc, q, &pf.normed, &pf.inj, t);
        }

        let gp = self.group_params(false, 0.0);
        let nbu = t as u32;

        self.dispatch(
            enc,
            &self.pipes.hc_mix_b,
            |e| {
                self.bind(e, 0, &pf.u, 0);
                self.bind(e, 1, &pf.normed, 0);
                self.bind(e, 2, &pf.mixed, 0);
                set_bytes(e, 3, &gp);
                set_bytes(e, 4, &nbu);
            },
            t * h as usize,
            256,
            false,
        );
    }
}
