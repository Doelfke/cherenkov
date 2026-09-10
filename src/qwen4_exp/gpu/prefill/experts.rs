//! Prefill expert grouping and event-protected streaming ring.

use super::*;

// Pool entries stay resident for decode; ring entries are reused
// only after the GPU signals that their previous batch is finished.
enum ExpertSource {
    Pool {
        rid: usize,
        buf: Buf,
        record_offset: usize,
        fetch: bool,
    },
    Ring(u32),
}

struct ExpertJob {
    expert: usize,
    csr_offset: usize,
    row_count: usize,
    source: ExpertSource,
    layout: crate::qwen4_exp::lowbit::Layout,
}

impl Gpu<'_> {
    /// Group prompt rows by expert and reserve the resident/ring destinations.
    /// This preserves expert order, stable usage ranking and CSR row order.
    fn prepare_expert_jobs(
        &mut self,
        record_layer: usize,
        t: usize,
        pf: &PrefillScratch,
    ) -> Result<Vec<ExpertJob>> {
        let c = &self.p.cfg;
        let k = c.num_experts_per_tok;
        let base_layout = crate::qwen4_exp::lowbit::Layout::four_bit(&self.p.manifest.experts);
        let miss_layout = self.low_bit_store.unwrap_or(base_layout);
        let n_record_layers = self.p.manifest.experts.layers;
        let idx = self.read_u32(&pf.topk_idx, t * k);
        let wts = self.read_f32(&pf.topk_w, t * k);
        // Rows per expert, in expert order.
        let mut lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); c.num_experts];

        for r in 0..t {
            for j in 0..k {
                lists[idx[r * k + j] as usize].push((r as u32, wts[r * k + j]));
            }
        }

        // The set keeps this layer's most-used experts (its share of the
        // budget); everything else streams through the ring.
        let budget = self.res.budget() / n_record_layers;
        let mut by_use: Vec<usize> = (0..c.num_experts)
            .filter(|&e| !lists[e].is_empty())
            .collect();

        by_use.sort_by_key(|&e| std::cmp::Reverse(lists[e].len()));

        let keep: std::collections::HashSet<usize> = by_use.iter().take(budget).copied().collect();
        let mut csr_rows: Vec<u32> = Vec::with_capacity(t * k);
        let mut csr_w: Vec<f32> = Vec::with_capacity(t * k);
        let mut jobs: Vec<ExpertJob> = Vec::new();
        self.step_no += 1;
        let mut ring_pos = 0usize;

        for (e, list) in lists.iter().enumerate() {
            if list.is_empty() {
                continue;
            }

            let csr_offset = csr_rows.len();

            for &(r, w) in list {
                csr_rows.push(r);
                csr_w.push(w);
            }

            let rid = self.record_id(record_layer, e as u32);
            // Preserve a resident record's precision. New kept records
            // use the pool's default; transient misses use --miss-experts.
            let source = if keep.contains(&e) || self.res.is_member(rid) {
                let fetch = !self
                    .res
                    .acquire(&self.ctx, &[rid], self.step_no)?
                    .is_empty();
                let (buf, record_offset) = self.res.buf(&self.ctx, rid)?;

                ExpertSource::Pool {
                    rid,
                    buf,
                    record_offset,
                    fetch,
                }
            } else {
                let slot = (ring_pos % RING) as u32;
                ring_pos += 1;

                ExpertSource::Ring(slot)
            };
            let layout = match &source {
                ExpertSource::Pool { .. } if self.res.kind(rid) == 0 => base_layout,
                _ => miss_layout,
            };

            jobs.push(ExpertJob {
                layout,
                expert: e,
                csr_offset,
                row_count: list.len(),
                source,
            });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                csr_rows.as_ptr(),
                pf.csr_rows.contents().cast::<u32>().as_ptr(),
                csr_rows.len(),
            );
            std::ptr::copy_nonoverlapping(
                csr_w.as_ptr(),
                pf.csr_w.contents().cast::<f32>().as_ptr(),
                csr_w.len(),
            );
        }

        Ok(jobs)
    }

    /// The MoE of one block over t rows: shared expert over all rows, then
    /// the routed experts one at a time (their tokens gathered, computed,
    /// scatter-added). The layer's most-used experts (up to the pool's
    /// per-layer share) go into the residency set and stay for decode;
    /// the rest stream through the ring. Returns (records fetched, bytes fetched, seconds
    /// waiting on ring slots, GPU seconds).
    pub(super) fn pf_experts(
        &mut self,
        moe: MoeRef,
        t: usize,
        pf: &PrefillScratch,
    ) -> Result<(usize, usize, f64, f64)> {
        let h = self.p.cfg.hidden_size as u32;
        let inter = self.p.manifest.experts.inter as u32;
        let stride = self.p.manifest.experts.record_stride as usize;
        let base_layout = crate::qwen4_exp::lowbit::Layout::four_bit(&self.p.manifest.experts);
        let miss_layout = self.low_bit_store.unwrap_or(base_layout);
        let jobs = self.prepare_expert_jobs(moe.record_layer, t, pf)?;
        let n_batches = jobs.len().div_ceil(GROUP);
        // GPU -> CPU "batch done" on `event`, CPU -> GPU "batch fetched"
        // on `event_cpu`; each has one writer, so values only grow.
        let base = self.event_base;
        self.event_base += n_batches as u64 + 1;
        let cbase = self.event_cpu_base;
        self.event_cpu_base += n_batches as u64 + 1;
        let event: &ProtocolObject<dyn objc2_metal::MTLEvent> =
            ProtocolObject::from_ref(&*self.event);
        let event_cpu: &ProtocolObject<dyn objc2_metal::MTLEvent> =
            ProtocolObject::from_ref(&*self.event_cpu);

        let cb = self.ctx.queue.commandBuffer().context("command buffer")?;
        let mut enc = cb.computeCommandEncoder().context("encoder")?;

        // Shared expert over all rows, into a zeroed block output.
        self.zero(&enc, &pf.moe_out, t as u32 * h);

        if !self.skips("shared") {
            self.qmm(&enc, &moe.sg, &pf.mixed, &pf.ge, t);
            self.qmm(&enc, &moe.su, &pf.mixed, &pf.ue, t);
            self.dispatch(
                &enc,
                &self.pipes.silu_mul,
                |e| {
                    self.bind(e, 0, &pf.ge, 0);
                    self.bind(e, 1, &pf.ue, 0);
                    self.bind(e, 2, &pf.hg, 0);
                },
                t * inter as usize,
                256,
                false,
            );
            self.qmm(&enc, &moe.sd, &pf.hg, &pf.ye, t);
            self.dispatch(
                &enc,
                &self.pipes.shared_add_rows,
                |e| {
                    self.bind(e, 0, &pf.moe_out, 0);
                    self.bind(e, 1, &pf.ye, 0);
                    self.bind(e, 2, &self.dense, moe.gate.0);
                    self.bind(e, 3, &pf.mixed, 0);
                    set_bytes(e, 4, &h);
                },
                t,
                256,
                true,
            );
        }

        if !self.skips("experts") {
            for (bi, batch) in jobs.chunks(GROUP).enumerate() {
                enc.endEncoding();
                cb.encodeWaitForEvent_value(event_cpu, cbase + bi as u64 + 1);

                enc = cb.computeCommandEncoder().context("encoder")?;

                for job in batch {
                    let (csr_offset, n) = (job.csr_offset, job.row_count);
                    let nu = n as u32;
                    let (wb, rec) = match &job.source {
                        ExpertSource::Pool {
                            buf, record_offset, ..
                        } => (buf, *record_offset),
                        ExpertSource::Ring(slot) => (&self.ring, *slot as usize * stride),
                    };

                    self.dispatch(
                        &enc,
                        &self.pipes.gather_rows,
                        |e| {
                            self.bind(e, 0, &pf.mixed, 0);
                            self.bind(e, 1, &pf.csr_rows, csr_offset * 4);
                            self.bind(e, 2, &pf.xg, 0);
                            set_bytes(e, 3, &nu);
                            set_bytes(e, 4, &h);
                        },
                        n * h as usize,
                        256,
                        false,
                    );

                    let l = job.layout;
                    let gate = Q {
                        w: l.gate_w,
                        s: l.gate_s,
                        b: l.gate_b,
                        out: inter,
                        inp: h,
                    };
                    let up = Q {
                        w: l.up_w,
                        s: l.up_s,
                        b: l.up_b,
                        out: inter,
                        inp: h,
                    };
                    let down = Q {
                        w: l.down_w,
                        s: l.down_s,
                        b: l.down_b,
                        out: h,
                        inp: inter,
                    };

                    self.expert_qmm_from(&enc, wb, &gate.at_offset(rec), &pf.xg, &pf.ge, n, l.bits);
                    self.expert_qmm_from(&enc, wb, &up.at_offset(rec), &pf.xg, &pf.ue, n, l.bits);
                    self.dispatch(
                        &enc,
                        &self.pipes.silu_mul,
                        |e| {
                            self.bind(e, 0, &pf.ge, 0);
                            self.bind(e, 1, &pf.ue, 0);
                            self.bind(e, 2, &pf.hg, 0);
                        },
                        n * inter as usize,
                        256,
                        false,
                    );
                    self.expert_qmm_from(&enc, wb, &down.at_offset(rec), &pf.hg, &pf.ye, n, l.bits);
                    self.dispatch(
                        &enc,
                        &self.pipes.scatter_add_rows,
                        |e| {
                            self.bind(e, 0, &pf.ye, 0);
                            self.bind(e, 1, &pf.csr_rows, csr_offset * 4);
                            self.bind(e, 2, &pf.csr_w, csr_offset * 4);
                            self.bind(e, 3, &pf.moe_out, 0);
                            set_bytes(e, 4, &nu);
                            set_bytes(e, 5, &h);
                        },
                        n * h as usize,
                        256,
                        false,
                    );
                }

                enc.endEncoding();
                cb.encodeSignalEvent_value(event, base + bi as u64 + 1);

                enc = cb.computeCommandEncoder().context("encoder")?;
            }
        }

        enc.endEncoding();
        cb.commit();

        // Stream the batches' records ahead of the GPU.
        let mut fetched = 0usize;
        let mut fetched_bytes = 0usize;
        let mut wait_s = 0.0f64;
        let ring_base = self.ring.contents().cast::<u8>().as_ptr() as usize;

        for (bi, batch) in jobs.chunks(GROUP).enumerate() {
            if bi >= RING / GROUP {
                // The ring slots this batch overwrites were last used at
                // most RING/GROUP batches ago; the GPU must be done there.
                let need = base + (bi - RING / GROUP) as u64 + 1;
                let t0 = std::time::Instant::now();

                while self.event.signaledValue() < need {
                    std::hint::spin_loop();
                }

                wait_s += t0.elapsed().as_secs_f64();
            }

            let (records, bytes) = self.read_expert_batch(
                moe.record_layer,
                batch,
                ring_base,
                stride,
                miss_layout.kind(),
            )?;
            fetched += records;
            fetched_bytes += bytes;

            self.event_cpu.setSignaledValue(cbase + bi as u64 + 1);
        }

        cb.waitUntilCompleted();

        Ok((
            fetched,
            fetched_bytes,
            wait_s,
            cb.GPUEndTime() - cb.GPUStartTime(),
        ))
    }

    /// Read one batch into its reserved destinations before publishing its event.
    fn read_expert_batch(
        &mut self,
        record_layer: usize,
        batch: &[ExpertJob],
        ring_base: usize,
        stride: usize,
        miss_kind: u8,
    ) -> Result<(usize, usize)> {
        let (mut to_set, mut ring4, mut ring_low) = (Vec::new(), Vec::new(), Vec::new());

        for job in batch {
            match &job.source {
                ExpertSource::Pool {
                    rid, fetch: true, ..
                } => to_set.push(*rid),
                ExpertSource::Ring(slot) => {
                    let reads = if job.layout.bits == 4 {
                        &mut ring4
                    } else {
                        &mut ring_low
                    };

                    reads.push((
                        ring_base + *slot as usize * stride,
                        self.record_id(record_layer, job.expert as u32) * job.layout.stride,
                        job.layout.stride,
                    ));
                }
                _ => {}
            }
        }

        let fetched = to_set.len() + ring4.len() + ring_low.len();
        let fetched_bytes = batch
            .iter()
            .filter(|job| !matches!(job.source, ExpertSource::Pool { fetch: false, .. }))
            .map(|job| job.layout.stride)
            .sum::<usize>();

        if !self.fake_experts {
            let (plan, _) = self.res.plan_reads(&to_set);

            plan.run(&self.pool_file, &self.pool_file_nocache);
            residency::fetch_into_slots(&self.pool_file_nocache, &ring4);

            if !ring_low.is_empty() {
                let file = self.res.store_file(&self.pool_file_nocache, miss_kind);

                residency::fetch_into_slots(file, &ring_low);
            }
        }

        self.res.finish(&self.ctx, &to_set)?;

        Ok((fetched, fetched_bytes))
    }
}
