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

        if let Some(out) = self.folded_mtp.take() {
            // The trunk step already ran the head over its rows with the
            // trunk's predictions, which are `next` for every accepted row.
            // The fold's last accepted row supplies the first draft and the
            // residual that seeds any chained passes.
            self.mtp_len = self.batch_pos + n;
            let seed = out[n - 1];

            if chain > 0 {
                drafts.push(seed);
            }

            if chain >= 2 {
                // Batch the remaining chained single-row passes into one
                // command buffer instead of issuing a separate CB for each:
                // every pass's token is the previous pass's GPU-side argmax,
                // so the chain stays on-device with no CPU round trip.
                let outs = self.mtp_batch_pass(seed, n, chain - 1)?;

                drafts.extend(outs);
                self.mtp_ms.push(t0.elapsed().as_secs_f64() * 1e3);

                return Ok(drafts);
            }

            // 0 or 1 draft: the fold alone supplies it; no chained passes.
            self.mtp_ms.push(t0.elapsed().as_secs_f64() * 1e3);

            return Ok(drafts);
        }

        // No fold (prefix warmup or the developer override): run the full
        // head over `n` rows, then chain single-row passes one at a time
        // (the chain is short here, so per-pass command buffers are fine).
        let mut token_rows: Vec<u32> = next.to_vec();
        let mut src_hyper_row = 0usize;
        let mut first = true;

        for c in 0..chain.max(1) {
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

    /// Run `n_passes` chained single-row MTP passes in ONE command buffer,
    /// replacing the per-pass `mtp_pass` round trips (each its own CB).
    ///
    /// Pass 0 seeds from `seed` (written by the caller to
    /// `scratch.ids[IDS_MTP_IN]` here before commit) and the fold's residual
    /// at row `n - 1` (`scratch.mtp_hyper[(n - 1) * hh * 4]`). Each later pass
    /// seeds from the previous pass's argmax (GPU-internal, read from
    /// `scratch.ids[MTP_CHAIN_TOKENS_BASE + c - 1]`) and from
    /// `scratch.mtp_hyper[0]` (the previous pass's own residual output, as in
    /// the per-pass path). Every pass's argmax is written by its `head_b` to
    /// `scratch.ids[MTP_CHAIN_TOKENS_BASE + c]`.
    ///
    /// Every pass still runs its own expert-slot handshake under a distinct
    /// event `seq`, so `service_block` runs once per pass in GPU order.
    /// Returns the argmax tokens of all `n_passes` passes.
    #[allow(clippy::excessive_nesting)]
    pub(super) fn mtp_batch_pass(
        &mut self,
        seed: u32,
        n: usize,
        n_passes: usize,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            (1..=MAX_NB - 1).contains(&n_passes),
            "chained passes must be 1..={}",
            MAX_NB - 1
        );

        unsafe {
            let ids = self.scratch.ids.contents().cast::<u32>().as_ptr();

            ids.add(IDS_MTP_IN).write(seed);
        }

        self.dispatch_count.set(0);

        // `step_no` is a per-pass LRU sequence (see
        // `residency::acquire`): each chained pass gets its own so that
        // pass i+1 evicts only records not touched by pass i, matching the
        // per-pass `mtp_pass` semantics exactly.
        let base_seq = self.event_base + 1;
        self.event_base += 4 * n_passes as u64;
        let phase_clock = self.phase_clock();

        let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
        let mut enc = self.phase_encoder(&cb, None, 4 * self.layers.len())?;
        let slot_row = self.layers.len();
        let hh = self.p.cfg.hc_hidden();

        // Pass 0 (the first chained pass after the fold) seeds the head's
        // residual from the fold's own output at row `n - 1` (the last
        // committed row's position, mirroring `mtp_pass(from_trunk=false,
        // src_row=n-1)`). Each subsequent pass seeds from row 0, which the
        // previous pass's `mtp_fold` kernel overwrote with its output.
        for c in 0..n_passes {
            let ids_in = if c == 0 {
                IDS_MTP_IN
            } else {
                MTP_CHAIN_TOKENS_BASE + c - 1
            };
            let residual_off = if c == 0 { (n - 1) * hh * 4 } else { 0 };
            let base_pos = self.batch_pos + n + c;

            anyhow::ensure!(base_pos < self.max_t, "context capacity exceeded");

            self.encode_mtp_prelude(
                &enc,
                self.mtp.as_ref().context("no MTP head")?,
                1,
                ids_in,
                &self.scratch.mtp_hyper,
                residual_off,
            );

            self.encode_block(
                &cb,
                &mut enc,
                &self.mtp.as_ref().context("no MTP head")?.layer,
                slot_row,
                base_pos,
                1,
                0,
                &self.scratch.mtp_hyper,
                None,
                None,
                base_seq + 4 * c as u64,
            )?;

            self.head_b(
                &enc,
                &self.mtp.as_ref().context("no MTP head")?.mixer,
                1,
                &self.scratch.mtp_hyper,
                0,
                Some(&self.scratch.moe_out),
                &self.scratch.mtp_logits2,
                MTP_CHAIN_TOKENS_BASE + c,
            );
        }

        enc.endEncoding();
        cb.commit();

        let record_layer = self
            .mtp
            .as_ref()
            .context("no MTP head")?
            .layer
            .moe
            .record_layer;
        let mut predicted = std::collections::VecDeque::new();
        let (mut io_s, mut turn_s) = (0.0f64, 0.0f64);

        for c in 0..n_passes {
            // Mirror the per-pass path: each pass gets a distinct LRU
            // sequence, so pass i+1 cannot evict records still needed by
            // pass i (see residency::acquire).
            self.step_no += 1;

            self.service_block(
                record_layer,
                slot_row,
                1,
                base_seq + 4 * c as u64,
                &mut predicted,
                None,
                &mut io_s,
                &mut turn_s,
            )?;
        }

        cb.waitUntilCompleted();
        self.collect_phases(&[(slot_row, record_layer)], phase_clock);

        let ids = self.scratch.ids.contents().cast::<u32>().as_ptr();
        let outs: Vec<u32> = (0..n_passes)
            .map(|c| unsafe { ids.add(MTP_CHAIN_TOKENS_BASE + c).read() })
            .collect();

        Ok(outs)
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
