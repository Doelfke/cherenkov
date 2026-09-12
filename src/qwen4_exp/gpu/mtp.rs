//! Draft-head execution and folded/chained MTP inputs.

use super::*;

impl Gpu<'_> {
    /// MTP draft head after a commit of `next.len()` rows: row r pairs the
    /// trunk's wide residual at batch_pos + r with `next[r]`, the token that
    /// follows it. Fills the head's KV cache for those positions and returns
    /// `chain` greedy drafts for the positions after the last committed
    /// token (each chained draft re-enters the head on its own residual).
    pub fn mtp_draft(&mut self, next: &[u32], chain: usize) -> Result<Vec<u32>> {
        let n = next.len();

        anyhow::ensure!(
            n >= 1 && n <= self.batch_nb,
            "MTP rows must match the committed rows"
        );
        anyhow::ensure!(
            self.mtp_len == self.batch_pos,
            "MTP cache out of step: {} vs {}",
            self.mtp_len,
            self.batch_pos
        );

        let t0 = std::time::Instant::now();
        let mut drafts = Vec::with_capacity(chain);
        let mut token_rows: Vec<u32> = next.to_vec();
        let mut src_hyper_row = 0usize;
        let mut first = true;
        let mut c0 = 0;

        if let Some(out) = self.folded_mtp.take() {
            // The trunk step already ran the head over its rows with the
            // trunk's predictions, which are `next` for every accepted row.
            self.mtp_len = self.batch_pos + n;
            src_hyper_row = n - 1;
            first = false;
            token_rows = vec![out[n - 1]];

            if chain > 0 {
                drafts.push(out[n - 1]);
            }

            c0 = 1;
        }

        for c in c0..chain.max(1) {
            let rows = if first { n } else { 1 };
            let base_pos = if first {
                self.batch_pos
            } else {
                self.batch_pos + n - 1 + c
            };

            anyhow::ensure!(base_pos + rows <= self.max_t, "context capacity exceeded");

            let out = self.mtp_pass(&token_rows, rows, base_pos, first, src_hyper_row)?;
            let d = out[rows - 1];

            if first {
                self.mtp_len = self.batch_pos + n;
                src_hyper_row = n - 1;
            } else {
                src_hyper_row = 0;
            }

            if c < chain {
                drafts.push(d);
            }

            token_rows = vec![d];
            first = false;
        }

        self.mtp_ms.push(t0.elapsed().as_secs_f64() * 1e3);

        Ok(drafts)
    }

    /// One MTP pass over `rows` rows: hidden input from the trunk's wide
    /// residual rows 0.. (`from_trunk`) or from the head's own residual row
    /// `src_row`; tokens in `tokens`; positions base_pos... Returns the
    /// argmax per row (logits stay in scratch.mtp_logits).
    pub(super) fn mtp_pass(
        &mut self,
        tokens: &[u32],
        rows: usize,
        base_pos: usize,
        from_trunk: bool,
        src_row: usize,
    ) -> Result<Vec<u32>> {
        let hh = self.p.cfg.hc_hidden();
        let phase_clock = self.phase_clock();
        let mtp = self.mtp.as_ref().context("no MTP head")?;

        unsafe {
            let ids = self.scratch.ids.contents().cast::<u32>().as_ptr();

            for (i, &t) in tokens.iter().enumerate() {
                ids.add(IDS_MTP_IN + i).write(t);
            }
        }

        self.dispatch_count.set(0);

        self.step_no += 1;
        let seq = self.event_base + 1;
        self.event_base += 4;
        let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
        let mut enc = self.phase_encoder(&cb, None, 4 * self.layers.len())?;
        let (src, off) = if from_trunk {
            (&self.scratch.hyper, 0)
        } else {
            (&self.scratch.mtp_hyper, src_row * hh * 4)
        };

        self.encode_mtp_prelude(&enc, mtp, rows, IDS_MTP_IN, src, off);

        let slot_row = self.layers.len();

        self.encode_block(
            &cb,
            &mut enc,
            &mtp.layer,
            slot_row,
            base_pos,
            rows,
            0,
            &self.scratch.mtp_hyper,
            None,
            None,
            seq,
        )?;

        let logits = if from_trunk {
            &self.scratch.mtp_logits
        } else {
            &self.scratch.mtp_logits2
        };

        self.head_b(
            &enc,
            &mtp.mixer,
            rows,
            &self.scratch.mtp_hyper,
            0,
            Some(&self.scratch.moe_out),
            logits,
            IDS_MTP_OUT,
        );
        enc.endEncoding();
        cb.commit();

        let record_layer = mtp.layer.moe.record_layer;
        let mut predicted = std::collections::VecDeque::new();
        let (mut io_s, mut turn_s) = (0.0, 0.0);

        self.service_block(
            record_layer,
            slot_row,
            rows,
            seq,
            &mut predicted,
            None,
            &mut io_s,
            &mut turn_s,
        )?;
        cb.waitUntilCompleted();
        self.collect_phases(&[(slot_row, record_layer)], phase_clock);

        Ok(self.read_u32(&self.scratch.ids, IDS_MTP_OUT + rows)[IDS_MTP_OUT..].to_vec())
    }

    /// The MTP head's input stream for `rows` rows: tokens from
    /// scratch.ids[ids_in..], residual rows from `src` (byte offset `off`),
    /// folded into scratch.mtp_hyper.
    pub(super) fn encode_mtp_prelude(
        &self,
        enc: &Enc,
        mtp: &Mtp,
        rows: usize,
        ids_in: usize,
        src: &Buf,
        off: usize,
    ) {
        let c = &self.p.cfg;
        let h = c.hidden_size as u32;
        let hc = c.hc_count;
        let hh = c.hc_hidden();
        let s = &self.scratch;
        let (i0, nbu) = (ids_in as u32, rows as u32);

        self.dispatch(
            enc,
            &self.pipes.embed_rows,
            |e| {
                self.bind(e, 0, &s.ids, 0);
                self.bind(e, 1, &self.dense, self.embed.w);
                self.bind(e, 2, &self.dense, self.embed.s);
                self.bind(e, 3, &self.dense, self.embed.b);
                self.bind(e, 4, &s.e, 0);
                set_bytes(e, 5, &h);
                set_bytes(e, 6, &i0);
                set_bytes(e, 7, &nbu);
            },
            rows * h as usize,
            256,
            false,
        );
        // e_norm = rmsnorm(e) * (1 + enorm)  ->  fe = fc_e(e_norm)
        self.group_norm_b(enc, &s.e, 0, mtp.enorm, &s.hc.mixed, h, 1, 1.0, rows);
        self.prep_h(enc, &s.hc.mixed, 0, h, rows, &s.hc.h1);
        self.qmv_h(enc, &mtp.fc_e, &s.fe, rows, &s.hc.h1);
        // Per stream: fh[b*hc+g] = fc_h(rmsnorm(src[b][g]) * (1 + hnorm[g]))
        self.group_norm_b(
            enc,
            src,
            off,
            mtp.hnorm,
            &s.hc.normed,
            h,
            hc as u32,
            1.0,
            rows,
        );
        self.prep_h(enc, &s.hc.normed, 0, h, rows * hc, &s.hc.h1);
        self.qmv_h(enc, &mtp.fc_h, &s.fh, rows * hc, &s.hc.h1);

        let gp = self.group_params(false, 0.0);

        self.dispatch(
            enc,
            &self.pipes.mtp_fold,
            |e| {
                self.bind(e, 0, &s.fe, 0);
                self.bind(e, 1, &s.fh, 0);
                self.bind(e, 2, &s.mtp_hyper, 0);
                set_bytes(e, 3, &gp);
                set_bytes(e, 4, &nbu);
            },
            rows * hh,
            256,
            false,
        );
    }

    /// A second draft chained on the MTP residual left by `prefill_chunk`.
    pub fn mtp_chain(&mut self, draft: u32) -> Result<u32> {
        let out = self.mtp_pass(&[draft], 1, self.pos, false, 0)?;

        Ok(out[0])
    }
}
