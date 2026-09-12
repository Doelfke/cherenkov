//! Prefill engine: a prompt chunk of up to a few thousand tokens runs
//! through the model layer by layer, so each layer's experts are streamed
//! once for the whole chunk instead of once per few tokens.
//!
//! Dense projections use simdgroup-matrix GEMM kernels;
//! attention is the GEMM form (S = QK^T, causal softmax with the QSA
//! visibility mask, O = PV) in sub-chunks of queries; the DeltaNet scan
//! and the PLE conv walk the chunk sequentially per channel. Routed
//! experts are handled per expert: its tokens' rows are gathered, run
//! through gate/up/act/down GEMMs and scatter-added back with their
//! routing weights; records stream through a ring of pool slots, a fetch
//! thread filling batches ahead of the GPU with two events per batch.
//! Every side effect (KV cache, index keys, recurrent state, PLE ring,
//! MTP cache) is left exactly as the row-batched engine would have left
//! it, so decode continues from the result.

use super::*;

pub(super) mod allocation;
mod attention;
mod deltanet;
mod experts;
mod hyperconnection;
mod ple;
mod projection;
mod stats;
pub use stats::PrefillStats;

/// Slots of the expert stream ring, and experts per fetch batch.
pub(super) const RING: usize = 64;
const GROUP: usize = 8;
/// Query rows per attention sub-chunk (bounds the S matrix).
const QS: usize = 256;

/// Timing of one prefill chunk.
#[derive(Clone, Copy, Default)]
pub struct ChunkStats {
    pub tokens: usize,
    pub secs: f64,
    /// Expert records read from disk (ring + pool).
    pub fetched: usize,
    /// Bytes read in each record's stored precision.
    pub fetched_bytes: usize,
    /// Seconds the fetch thread waited for ring slots.
    pub wait_s: f64,
    /// GPU seconds: DeltaNet blocks, attention blocks, expert streams, MTP.
    pub gpu_delta_s: f64,
    pub gpu_attn_s: f64,
    pub gpu_experts_s: f64,
    pub gpu_mtp_s: f64,
    /// CPU seconds gathering n-gram rows (page faults included).
    pub ngram_s: f64,
}

#[derive(Clone, Copy)]
struct MoeRef {
    record_layer: usize,
    sg: Q,
    su: Q,
    sd: Q,
    gate: T,
}

pub(super) struct PrefillScratch<T = Buf> {
    allocated_bytes: usize,
    rows: usize,
    ids: T,
    e: T,
    hyper: T,
    normed: T,
    d: T,
    u: T,
    mixed: T,
    inj: T,
    mix_out: T,
    moe_out: T,
    qg: T,
    k: T,
    v: T,
    attn_out: T,
    iqk: T,
    iq: T,
    bscore: T,
    vis: T,
    nvis: T,
    vmask: T,
    ag_qh: T,
    ag_kh: T,
    ag_vt: T,
    ag_s: T,
    ag_p: T,
    ag_o: T,
    qkv: T,
    z: T,
    a: T,
    b: T,
    kqn: T,
    gbuf: T,
    delta_y: T,
    router: T,
    topk_idx: T,
    topk_w: T,
    csr_rows: T,
    csr_w: T,
    xg: T,
    ge: T,
    ue: T,
    hg: T,
    ye: T,
    mtp_hyper: T,
    fe: T,
    fh: T,
    logits_all: Option<T>,
}

pub const MAX_PREFILL_ROWS: usize = 4096;

impl<'a> Gpu<'a> {
    /// One decoder block over t rows of `hyper`: everything up to the
    /// router in one command buffer, then the expert stream. Returns
    /// (records fetched, bytes fetched, ring wait seconds, block GPU seconds, expert
    /// stream GPU seconds).
    #[allow(clippy::too_many_arguments)]
    fn pf_layer(
        &mut self,
        li: Option<usize>,
        base: usize,
        t: usize,
        pf: &PrefillScratch,
        hyper: &Buf,
        pending: Option<&Buf>,
    ) -> Result<(usize, usize, f64, f64, f64)> {
        let (moe, block_s) = {
            let layer = match li {
                Some(li) => &self.layers[li],
                None => &self.mtp.as_ref().context("no MTP head")?.layer,
            };
            let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
            let enc = cb.computeCommandEncoder().context("encoder")?;
            let mut pending = pending;

            if let Some(pl) = &layer.ple {
                if let Some(out) = pending.take() {
                    self.pf_inject(&enc, hyper, out, &pf.inj, t);
                }

                self.pf_ple(&enc, pl, base, t, pf)?;
            }

            self.pf_hc_read(&enc, &layer.attn_hc, t, pf, hyper, pending, true);

            if !self.skips("mixer") {
                match &layer.mix {
                    Mix::Attn(a) => self.pf_attention(&enc, a, base, t, pf),
                    Mix::Delta(d) => self.pf_deltanet(&enc, d, t, pf),
                }
            }

            self.pf_hc_read(&enc, &layer.mlp_hc, t, pf, hyper, Some(&pf.mix_out), true);
            self.router_b(
                &enc,
                &layer.moe,
                t,
                &pf.mixed,
                &pf.router,
                &pf.topk_idx,
                &pf.topk_w,
                self.p.cfg.num_experts_per_tok,
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();

            (
                MoeRef {
                    record_layer: layer.moe.record_layer,
                    sg: layer.moe.sg,
                    su: layer.moe.su,
                    sd: layer.moe.sd,
                    gate: layer.moe.shared_gate,
                },
                cb.GPUEndTime() - cb.GPUStartTime(),
            )
        };
        let (fetched, fetched_bytes, wait_s, experts_s) = self.pf_experts(moe, t, pf)?;

        Ok((fetched, fetched_bytes, wait_s, block_s, experts_s))
    }

    /// Run one prompt chunk at positions pos.. through the trunk and the
    /// MTP head, committing everything. `next_after` is the token that
    /// follows the chunk when known (the next chunk's first token); the
    /// trunk's own prediction pairs with the last row otherwise. With
    /// `all_logits` every row's logits are kept for checks. Returns the
    /// greedy token after the chunk and the MTP's first draft.
    pub fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        next_after: Option<u32>,
        all_logits: bool,
    ) -> Result<(u32, u32)> {
        let t = tokens.len();

        anyhow::ensure!(t >= 1, "empty chunk");
        anyhow::ensure!(self.pos + t <= self.max_t, "context capacity exceeded");

        let t0 = std::time::Instant::now();

        if self
            .pf
            .as_ref()
            .is_none_or(|pf| pf.rows < t || (all_logits && pf.logits_all.is_none()))
        {
            self.pf = None;
            self.pf = Some(self.pf_alloc(t, all_logits)?);
        }

        let pf = self.pf.take().unwrap();
        let c = &self.p.cfg;
        let h = c.hidden_size as u32;
        let hh = c.hc_hidden();
        let pos = self.pos;

        self.tokens.truncate(pos);
        self.tokens.extend_from_slice(tokens);
        self.ngram_prefetch_start(pos, t);

        let write_ids = |buf: &Buf, ids: &[u32]| unsafe {
            std::ptr::copy_nonoverlapping(
                ids.as_ptr(),
                buf.contents().cast::<u32>().as_ptr(),
                ids.len(),
            );
        };

        write_ids(&pf.ids, tokens);

        let mut st = ChunkStats {
            tokens: t,
            ..Default::default()
        };
        let ngram0 = self.ngram_gather_s.get();

        // Embedding into the replicated stream.
        {
            let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
            let enc = cb.computeCommandEncoder().context("encoder")?;
            let (i0, nbu) = (0u32, t as u32);

            self.dispatch(
                &enc,
                &self.pipes.embed_rows,
                |e| {
                    self.bind(e, 0, &pf.ids, 0);
                    self.bind(e, 1, &self.dense, self.embed.w);
                    self.bind(e, 2, &self.dense, self.embed.s);
                    self.bind(e, 3, &self.dense, self.embed.b);
                    self.bind(e, 4, &pf.e, 0);
                    set_bytes(e, 5, &h);
                    set_bytes(e, 6, &i0);
                    set_bytes(e, 7, &nbu);
                },
                t * h as usize,
                256,
                false,
            );

            let gp = self.group_params(false, 0.0);

            self.dispatch(
                &enc,
                &self.pipes.replicate_b,
                |e| {
                    self.bind(e, 0, &pf.e, 0);
                    self.bind(e, 1, &pf.hyper, 0);
                    set_bytes(e, 2, &gp);
                    set_bytes(e, 3, &nbu);
                },
                t * hh,
                256,
                false,
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }

        let n_layers = self.layers.len().min(layer_cap());

        for li in 0..n_layers {
            let pending = if li > 0 { Some(&pf.moe_out) } else { None };
            let (f, bytes, w, block_s, experts_s) =
                self.pf_layer(Some(li), pos, t, &pf, &pf.hyper, pending)?;
            st.fetched += f;
            st.fetched_bytes += bytes;
            st.wait_s += w;
            st.gpu_experts_s += experts_s;

            if matches!(self.layers[li].mix, Mix::Delta(_)) {
                st.gpu_delta_s += block_s;
            } else {
                st.gpu_attn_s += block_s;
            }
        }

        // Final injection, then the LM head on the last row (all rows when
        // checking).
        let cur = {
            let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
            let enc = cb.computeCommandEncoder().context("encoder")?;

            self.pf_inject(&enc, &pf.hyper, &pf.moe_out, &pf.inj, t);

            if let Some(la) = &pf.logits_all {
                self.pf_hc_read(&enc, &self.final_mixer, t, &pf, &pf.hyper, None, false);
                self.qmm(&enc, &self.lm_head, &pf.mixed, la, t);
            }

            self.head_b(
                &enc,
                &self.final_mixer,
                1,
                &pf.hyper,
                (t - 1) * hh * 4,
                None,
                &self.scratch.logits,
                IDS_OUT,
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();

            self.read_u32(&self.scratch.ids, IDS_OUT + 1)[IDS_OUT]
        };
        // MTP head over the chunk: row r pairs the trunk residual at pos+r
        // with the token after it.
        let mut draft = cur;

        if let Some(mtp) = self.mtp.as_ref() {
            let (enorm, hnorm, fc_e, fc_h) = (mtp.enorm, mtp.hnorm, mtp.fc_e, mtp.fc_h);
            let mut next: Vec<u32> = tokens[1..].to_vec();

            next.push(next_after.unwrap_or(cur));
            write_ids(&pf.ids, &next);

            {
                let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
                let enc = cb.computeCommandEncoder().context("encoder")?;
                let (i0, nbu) = (0u32, t as u32);

                self.dispatch(
                    &enc,
                    &self.pipes.embed_rows,
                    |e| {
                        self.bind(e, 0, &pf.ids, 0);
                        self.bind(e, 1, &self.dense, self.embed.w);
                        self.bind(e, 2, &self.dense, self.embed.s);
                        self.bind(e, 3, &self.dense, self.embed.b);
                        self.bind(e, 4, &pf.e, 0);
                        set_bytes(e, 5, &h);
                        set_bytes(e, 6, &i0);
                        set_bytes(e, 7, &nbu);
                    },
                    t * h as usize,
                    256,
                    false,
                );
                self.group_norm_b(&enc, &pf.e, 0, enorm, &pf.mixed, h, 1, 1.0, t);
                self.qmm(&enc, &fc_e, &pf.mixed, &pf.fe, t);
                self.group_norm_b(
                    &enc,
                    &pf.hyper,
                    0,
                    hnorm,
                    &pf.normed,
                    h,
                    c.hc_count as u32,
                    1.0,
                    t,
                );
                self.qmm(&enc, &fc_h, &pf.normed, &pf.fh, t * c.hc_count);

                let gp = self.group_params(false, 0.0);

                self.dispatch(
                    &enc,
                    &self.pipes.mtp_fold,
                    |e| {
                        self.bind(e, 0, &pf.fe, 0);
                        self.bind(e, 1, &pf.fh, 0);
                        self.bind(e, 2, &pf.mtp_hyper, 0);
                        set_bytes(e, 3, &gp);
                        set_bytes(e, 4, &nbu);
                    },
                    t * hh,
                    256,
                    false,
                );
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
            }

            let (f, bytes, w, block_s, experts_s) =
                self.pf_layer(None, pos, t, &pf, &pf.mtp_hyper, None)?;
            st.fetched += f;
            st.fetched_bytes += bytes;
            st.wait_s += w;
            st.gpu_mtp_s += block_s + experts_s;
            let mtp = self.mtp.as_ref().unwrap();
            let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
            let enc = cb.computeCommandEncoder().context("encoder")?;

            self.pf_inject(&enc, &pf.mtp_hyper, &pf.moe_out, &pf.inj, t);
            self.head_b(
                &enc,
                &mtp.mixer,
                1,
                &pf.mtp_hyper,
                (t - 1) * hh * 4,
                None,
                &self.scratch.mtp_logits,
                IDS_MTP_OUT,
            );

            // Keep the last row's residual where chained drafts re-enter.
            let n = hh as u32;

            self.dispatch(
                &enc,
                &self.pipes.copy_f32,
                |e| {
                    self.bind(e, 0, &pf.mtp_hyper, (t - 1) * hh * 4);
                    self.bind(e, 1, &self.scratch.mtp_hyper, 0);
                    set_bytes(e, 2, &n);
                },
                hh,
                256,
                false,
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();

            draft = self.read_u32(&self.scratch.ids, IDS_MTP_OUT + 1)[IDS_MTP_OUT];
            self.mtp_len = pos + t;
        }

        self.pos = pos + t;
        self.batch_pos = self.pos;
        self.batch_nb = 0;

        self.join_pending()?;

        st.secs = t0.elapsed().as_secs_f64();
        st.ngram_s = self.ngram_gather_s.get() - ngram0;

        self.activity.prefill.record(st);
        self.prefill_stats.push(st);

        self.pf = Some(pf);

        Ok((cur, draft))
    }

    /// Release the prefill scratch (a few hundred MB to GB).
    pub fn prefill_release(&mut self) {
        self.pf = None;
    }

    /// Debug: blocks selected for row `r` of the last prefill chunk (its
    /// last attention layer), from the block bitmask.
    pub fn debug_engine_blocks(&self, r: usize) -> Vec<u32> {
        let c = &self.p.cfg;
        let max_blocks = self.max_t / c.indexer_compress_ratio + 1;
        let words = max_blocks.div_ceil(32);
        let pf = self.pf.as_ref().expect("prefill scratch released");
        let m = self.read_u32(&pf.vmask, (r + 1) * words);

        (0..max_blocks as u32)
            .filter(|&j| (m[r * words + (j / 32) as usize] >> (j % 32)) & 1 == 1)
            .collect()
    }

    /// Debug: visible tokens of row `r` of the last row-batched step (its
    /// last attention layer).
    pub fn debug_row_vis(&self, r: usize) -> Vec<u32> {
        let c = &self.p.cfg;
        let stride = c.indexer_budget + c.indexer_compress_ratio;
        let n = self.read_u32(&self.scratch.nvis, r + 1)[r] as usize;

        self.read_u32(&self.scratch.vis, (r + 1) * stride)[r * stride..r * stride + n].to_vec()
    }

    /// Row `r` of the last prefill chunk's logits (all-logits mode).
    pub fn pf_logits_row(&self, r: usize) -> &[f32] {
        let v = self.p.cfg.vocab_size;
        let la = self
            .pf
            .as_ref()
            .and_then(|pf| pf.logits_all.as_ref())
            .expect("prefill logits not kept");
        let ptr = la.contents().cast::<f32>();

        unsafe { std::slice::from_raw_parts(ptr.as_ptr().add(r * v), v) }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/qwen4_exp/gpu/prefill.rs"]
mod tests;
