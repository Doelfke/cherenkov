//! Prefill PLE n-gram gather, gating, and dilated convolution.

use super::*;

impl Gpu<'_> {
    /// PLE block over t rows. Nothing else is live between blocks, so its
    /// buffers alias the stream-wide scratch: e -> mix_out, key -> normed
    /// (and the output, once the key is normed), keyn -> u, value -> mixed,
    /// query -> qkv, gated -> fh, gvn -> mtp_hyper.
    pub(super) fn pf_ple(
        &self,
        enc: &Enc,
        pl: &Ple,
        base: usize,
        t: usize,
        pf: &PrefillScratch,
    ) -> Result<()> {
        let c = &self.p.cfg;
        let h = c.hidden_size as u32;
        let hh = c.hc_hidden();
        anyhow::ensure!(
            c.ple_embed_dim <= c.hidden_size,
            "PLE embedding wider than the hidden size"
        );
        let (ple_e, ple_key, ple_keyn, ple_value, ple_query, ple_gated, ple_gvn, ple_out) = (
            &pf.mix_out,
            &pf.normed,
            &pf.u,
            &pf.mixed,
            &pf.qkv,
            &pf.fh,
            &pf.mtp_hyper,
            &pf.normed,
        );
        let dim = self.p.manifest.ngram.dim;
        let e = unsafe {
            std::slice::from_raw_parts_mut(
                ple_e.contents().cast::<f32>().as_ptr(),
                pf.rows * c.ple_embed_dim,
            )
        };
        let t_gather = std::time::Instant::now();
        self.ngram_prefetch_join();
        for b in 0..t {
            let ids = self.ngram_ids_at(pl, base + b);
            anyhow::ensure!(
                ids.len() * dim == c.ple_embed_dim,
                "n-gram head layout mismatch"
            );
            let eb = &mut e[b * c.ple_embed_dim..(b + 1) * c.ple_embed_dim];
            for (hi, &id) in ids.iter().enumerate() {
                self.p.ngram_row(id, &mut eb[hi * dim..(hi + 1) * dim]);
            }
        }
        self.ngram_gather_s
            .set(self.ngram_gather_s.get() + t_gather.elapsed().as_secs_f64());
        self.qmm(enc, &pl.key, ple_e, ple_key, t);
        self.qmm(enc, &pl.value, ple_e, ple_value, t);
        let groups = c.hc_count as u32;
        self.group_norm_b(enc, ple_key, 0, pl.norm_key, ple_keyn, h, groups, 0.0, t);
        self.group_norm_b(
            enc,
            &pf.hyper,
            0,
            pl.norm_query,
            ple_query,
            h,
            groups,
            0.0,
            t,
        );
        let gp = self.group_params(true, 0.0);
        self.dispatch(
            enc,
            &self.pipes.ple_gate_b,
            |e| {
                self.bind(e, 0, ple_keyn, 0);
                self.bind(e, 1, ple_query, 0);
                self.bind(e, 2, ple_value, 0);
                self.bind(e, 3, ple_gated, 0);
                set_bytes(e, 4, &gp);
            },
            t * c.hc_count,
            256,
            true,
        );
        self.group_norm_b(enc, ple_gated, 0, pl.norm_conv, ple_gvn, h, groups, 0.0, t);
        let cp = PleConvParams {
            channels: hh as u32,
            ksize: pl.kernel,
            dilation: pl.dilation,
            span: pl.span,
            filled: base as u32,
            nb: t as u32,
        };
        self.dispatch(
            enc,
            &self.pipes.ple_conv_b,
            |e| {
                self.bind(e, 0, ple_gated, 0);
                self.bind(e, 1, ple_gvn, 0);
                self.bind(e, 2, &self.dense, pl.conv.0);
                self.bind(e, 3, &pl.hist, 0);
                self.bind(e, 4, ple_out, 0);
                set_bytes(e, 5, &cp);
            },
            hh,
            256,
            false,
        );
        self.add(enc, &pf.hyper, ple_out, t * hh);
        Ok(())
    }
}
