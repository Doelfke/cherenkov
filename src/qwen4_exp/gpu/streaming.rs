//! Decode expert streaming: GPU routing, CPU slot publication, and file IO.
//!
//! The router and fetched-part signals share `event`; resident completion
//! uses `event_res`. Event waits protect the CPU/GPU ownership transitions
//! of the routing scratch, slot table, and expert records.

use super::*;

/// A background read may be absent under the FAKE developer option, but
/// its reserved records must still pass through residency completion.
pub(super) struct PendingRead {
    pub(super) thread: Option<std::thread::JoinHandle<()>>,
    pub(super) records: Vec<usize>,
}

/// One block's monotonic handshake. The two event objects have distinct
/// roles: do not publish `resident_done` on the router/table event.
struct BlockSignals {
    /// GPU -> CPU on event: routing scratch is readable.
    router_ready: u64,
    /// CPU -> GPU on event: table published; resident experts may run.
    resident_ready: u64,
    /// GPU -> CPU on event_res: optional weak-miss deadline boundary.
    resident_done: u64,
    /// CPU -> GPU on event: required misses are readable and resident.
    misses_ready: u64,
}

impl BlockSignals {
    fn new(seq: u64) -> Self {
        Self {
            router_ready: seq,
            resident_ready: seq + 1,
            resident_done: seq + 2,
            misses_ready: seq + 3,
        }
    }
}

impl Gpu<'_> {
    /// Wait for reads the deadline policy stopped waiting on, then make
    /// their records usable (nothing reuses their slots before this).
    pub(super) fn join_inflight(&mut self) -> Result<()> {
        if self.inflight.is_empty() {
            return Ok(());
        }
        let list = std::mem::take(&mut self.inflight);
        let rids: Vec<usize> = list.iter().map(|(_, r)| *r).collect();
        for (l, _) in &list {
            l.wait()?;
        }
        self.res.finish(&self.ctx, &rids)
    }

    /// Join the background read of predicted records and add them to the
    /// set (they are in memory once the read is done).
    pub(super) fn join_pending(&mut self) -> Result<()> {
        let Some(read) = self.pending.take() else {
            return Ok(());
        };
        let t = std::time::Instant::now();
        if let Some(thread) = read.thread {
            let _ = thread.join();
        }
        self.step_read_s += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        self.res.finish(&self.ctx, &read.records)?;
        self.step_set_s += t.elapsed().as_secs_f64();
        Ok(())
    }

    /// Read predicted records for the next layer on a background thread
    /// while the GPU runs the current layer's experts; they join the set
    /// at the next service. Returns records read.
    fn prefetch_async(&mut self, record_layer: usize, ids: &[u32]) -> Result<usize> {
        self.join_pending()?;
        let rids: Vec<usize> = ids
            .iter()
            .map(|&e| self.record_id(record_layer, e))
            .collect();
        let t_acq = std::time::Instant::now();
        let need = self.res.acquire(&self.ctx, &rids, self.step_no)?;
        self.step_set_s += t_acq.elapsed().as_secs_f64();
        if need.is_empty() {
            return Ok(0);
        }
        let (plan, warm) = self.res.plan_reads(&need);
        self.step_warm += warm;
        let n = plan.len();
        if self.fake_experts {
            self.pending = Some(PendingRead {
                thread: None,
                records: need,
            });
            return Ok(n);
        }
        let (cached, nocache) = (
            self.pool_file.try_clone().expect("dup experts fd"),
            self.pool_file_nocache.try_clone().expect("dup experts fd"),
        );
        self.pending = Some(PendingRead {
            thread: Some(std::thread::spawn(move || plan.run(&cached, &nocache))),
            records: need,
        });
        Ok(n)
    }

    /// Wait until the GPU releases the routing scratch for CPU reads.
    fn wait_for_router(&self, ready: u64, slot_row: usize) -> Result<()> {
        if !self.spin_wait {
            // Blocking avoids heating a CPU core on the fanless target machine.
            let ok = self.event.waitUntilSignaledValue_timeoutMS(ready, 30_000);
            anyhow::ensure!(
                ok,
                "GPU did not reach the router of slot row {slot_row} within 30 s"
            );
            return Ok(());
        }
        let started = std::time::Instant::now();
        while self.event.signaledValue() < ready {
            std::hint::spin_loop();
            anyhow::ensure!(
                started.elapsed().as_secs() <= 30,
                "GPU did not reach the router of slot row {slot_row} within 30 s"
            );
        }
        Ok(())
    }

    fn log_lookahead(
        &mut self,
        record_layer: usize,
        slot_row: usize,
        experts: &[u32],
        indices: &[u32],
        weights: &[f32],
    ) {
        if !self.log_la {
            return;
        }
        let k = self.p.cfg.num_experts_per_tok;
        for &expert in experts {
            let (mut weight, mut rank) = (0.0f32, u32::MAX);
            for (i, &selected) in indices.iter().enumerate() {
                if selected != expert {
                    continue;
                }
                weight = weight.max(weights[i]);
                rank = rank.min((i % k) as u32);
            }
            let resident = self.res.is_member(self.record_id(record_layer, expert));
            self.la_pending.push(LaEntry {
                step: self.step_no,
                layer: slot_row,
                expert,
                weight,
                rank,
                resident,
                hit: false,
            });
        }
    }

    /// One decoder block (attention or DeltaNet, then MoE) over `nb` rows
    /// of `hyper`, with the event handshake around the expert dispatch.
    /// The block's MoE output is left in scratch.moe_out for the next
    /// fused norm to inject.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_block(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        enc: &mut Retained<Enc>,
        layer: &GLayer,
        slot_row: usize,
        base_pos: usize,
        nb: usize,
        snap_after: usize,
        hyper: &Buf,
        pending: Option<&Buf>,
        lookahead: Option<&GLayer>,
        seq: u64,
    ) -> Result<()> {
        let signals = BlockSignals::new(seq);
        let s = &self.scratch;
        let event: &ProtocolObject<dyn objc2_metal::MTLEvent> =
            ProtocolObject::from_ref(&*self.event);
        self.hc_read_b(enc, &layer.attn_hc, nb, hyper, 0, pending, &s.hc, true);
        if !self.skips("mixer") {
            match &layer.mix {
                Mix::Attn(a) => self.attention_b(enc, a, base_pos, nb),
                Mix::Delta(d) => self.deltanet_b(enc, d, nb, snap_after),
            }
        }
        self.hc_read_b(
            enc,
            &layer.mlp_hc,
            nb,
            hyper,
            0,
            Some(&s.mix_out),
            &s.hc,
            true,
        );
        self.router_b(
            enc,
            &layer.moe,
            nb,
            &s.hc.mixed,
            &s.router,
            &s.topk_idx,
            &s.topk_w,
            self.p.cfg.num_experts_per_tok,
        );
        if let Some(next) = lookahead {
            // Approximate the next layer's routing on the current stream.
            self.hc_read_b(enc, &next.mlp_hc, nb, hyper, 0, None, &s.la, false);
            self.router_b(
                enc,
                &next.moe,
                nb,
                &s.la.mixed,
                &s.la_router,
                &s.la_idx,
                &s.la_w,
                self.p.cfg.num_experts_per_tok,
            );
        }
        enc.endEncoding();
        cb.encodeSignalEvent_value(event, signals.router_ready);
        cb.encodeWaitForEvent_value(event, signals.resident_ready);
        *enc = cb.computeCommandEncoder().context("encoder")?;
        self.experts_b(enc, &layer.moe, slot_row, nb, 0);
        enc.endEncoding();
        let event_res: &ProtocolObject<dyn objc2_metal::MTLEvent> =
            ProtocolObject::from_ref(&*self.event_res);
        cb.encodeSignalEvent_value(event_res, signals.resident_done);
        cb.encodeWaitForEvent_value(event, signals.misses_ready);
        *enc = cb.computeCommandEncoder().context("encoder")?;
        self.experts_b(enc, &layer.moe, slot_row, nb, 1);
        Ok(())
    }

    /// Fetched records use the miss precision; resident records keep their kind.
    fn miss_record_bytes(&mut self, need: &[usize]) -> usize {
        let Some(store) = self.low_bit_store else {
            return self.p.manifest.experts.record_stride as usize;
        };
        if !self.all_low_bits {
            let kind = if store.bits == 2 { 2 } else { 1 };
            for &rid in need {
                self.res.set_kind(rid, kind);
            }
        }
        store.stride
    }

    /// Wait for required reads and truncate late weak records at the deadline.
    /// The caller has published the resident table and has not released misses.
    #[allow(clippy::too_many_arguments)]
    fn wait_for_deadline(
        &mut self,
        slot_row: usize,
        resident_done: u64,
        flags: &[residency::Landed],
        need: &[usize],
        rids: &[usize],
        order: &[usize],
        n_res: usize,
        wmax: &[f32],
    ) -> Result<()> {
        let t = std::time::Instant::now();
        let landed = |f: &residency::Landed| f.done();
        // Whichever comes first: every read landed (release at once, as
        // before), or the GPU finished the resident part (cut the weak
        // stragglers, wait for the strong ones).
        let mut cut_now = false;
        loop {
            if flags.iter().all(landed) {
                break;
            }
            if self.event_res.signaledValue() >= resident_done {
                cut_now = true;
                break;
            }
            std::hint::spin_loop();
            if t.elapsed().as_secs() > 30 {
                anyhow::bail!(
                    "GPU did not finish the resident experts of slot row {slot_row} within 30 s"
                );
            }
        }
        if cut_now {
            let need_pos = |rid: usize| need.iter().position(|&r| r == rid).unwrap();
            let mut active_count = order.len();
            // Late entries sit at order[n_res..], strongest first.
            let mut u = order.len();
            while u > n_res {
                let i = order[u - 1];
                let f = &flags[need_pos(rids[i])];
                if landed(f) {
                    break;
                }
                if wmax[i] < self.cut_w {
                    active_count -= 1;
                    self.step_cut += 1;
                    self.inflight.push((f.clone(), rids[i]));
                    u -= 1;
                } else {
                    break;
                }
            }
            for &i in &order[n_res..active_count] {
                flags[need_pos(rids[i])].wait()?;
            }
            unsafe {
                let tab = self
                    .slot_tab
                    .contents()
                    .cast::<u64>()
                    .as_ptr()
                    .add(slot_row * SLOT_STRIDE);
                tab.add(SLOT_STRIDE - 1).write(active_count as u64);
            }
        }
        self.step_read_s += t.elapsed().as_secs_f64();
        Ok(())
    }

    /// Cut reads stay in flight until step end; publish only completed records.
    fn finish_miss_reads(
        &mut self,
        need: &[usize],
        flags: &[residency::Landed],
        deadline: bool,
    ) -> Result<()> {
        if !deadline {
            return self.res.finish(&self.ctx, need);
        }
        let landed: Vec<usize> = need
            .iter()
            .enumerate()
            .filter(|(j, _)| flags.is_empty() || flags[*j].done())
            .map(|(_, &r)| r)
            .collect();
        self.res.finish(&self.ctx, &landed)
    }

    /// CPU side of one block's handshake: wait for the router, publish the
    /// union of the rows' experts with the resident ones first. Read misses
    /// while the GPU computes the resident part, then issue lookahead and
    /// release the fetched part. Returns (lookahead hits, total experts,
    /// records read for lookahead). See BlockSignals for the event contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn service_block(
        &mut self,
        record_layer: usize,
        slot_row: usize,
        nb: usize,
        seq: u64,
        predicted: &mut std::collections::VecDeque<(usize, Vec<u32>)>,
        lookahead: Option<usize>,
        io_s: &mut f64,
        turn_s: &mut f64,
    ) -> Result<(usize, usize, usize)> {
        let signals = BlockSignals::new(seq);
        let k = self.p.cfg.num_experts_per_tok;
        self.wait_for_router(signals.router_ready, slot_row)?;
        let t0 = std::time::Instant::now();
        let idx = self.read_u32(&self.scratch.topk_idx, nb * k);
        let wts = self.read_f32(&self.scratch.topk_w, nb * k);
        if self.dump_states {
            // The GPU is stalled on the union, so this block's router
            // input is still in scratch.
            let h = self.p.cfg.hidden_size;
            self.last_states
                .push((self.read_f32(&self.scratch.hc.mixed, h), idx[..k].to_vec()));
        }
        let union = union_of(&idx);
        let (mut hits, mut total) = (0, 0);
        if predicted.front().is_some_and(|(row, _)| *row == slot_row) {
            let (_, pred) = predicted.pop_front().unwrap();
            hits = union.iter().filter(|e| pred.contains(e)).count();
            total = union.len();
        }
        for mut e in self.la_pending.drain(..) {
            e.hit = union.contains(&e.expert);
            self.la_log.push(e);
        }
        // Records the previous layer read for this one join the set now.
        self.join_pending()?;
        let rids: Vec<usize> = union
            .iter()
            .map(|&e| self.record_id(record_layer, e))
            .collect();
        let t_acq = std::time::Instant::now();
        let need = self.res.acquire(&self.ctx, &rids, self.step_no)?;
        self.step_set_s += t_acq.elapsed().as_secs_f64();
        self.step_misses += need.len();
        let miss_bytes = self.miss_record_bytes(&need);
        self.step_miss_bytes += need.len() * miss_bytes;
        // Start reading the misses at once, on their own thread; the GPU
        // runs the resident experts meanwhile.
        let ti = std::time::Instant::now();
        let (plan, warm) = self.res.plan_reads(&need);
        self.step_warm += warm;
        let deadline = self.cut_w > 0.0 && !self.fake_experts;
        // Per-record completion only for the deadline policy; otherwise
        // one thread for the batch. Metal IO lost at this read fan-out.
        let mut flags: Vec<residency::Landed> = Vec::new();
        let miss_read = if plan.is_empty() || self.fake_experts {
            None
        } else if deadline {
            flags = plan.run_tracked(&self.pool_file, &self.pool_file_nocache, need.len());
            None
        } else {
            let (cached, nocache) = (
                self.pool_file.try_clone().expect("dup experts fd"),
                self.pool_file_nocache.try_clone().expect("dup experts fd"),
            );
            Some(std::thread::spawn(move || plan.run(&cached, &nocache)))
        };
        // Resident records first, then the ones being fetched, strongest
        // first (so a deadline cut is a truncation of the weak end).
        let missing: Vec<bool> = rids.iter().map(|r| need.contains(r)).collect();
        let wmax: Vec<f32> = union
            .iter()
            .map(|&e| {
                (0..nb * k)
                    .filter(|&i| idx[i] == e)
                    .map(|i| wts[i])
                    .fold(0.0f32, f32::max)
            })
            .collect();
        let mut order: Vec<usize> = (0..union.len()).collect();
        order.sort_by(|&a, &b| {
            missing[a].cmp(&missing[b]).then(
                wmax[b]
                    .partial_cmp(&wmax[a])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        let n_res = missing.iter().filter(|m| !**m).count();
        let addrs: Vec<u64> = order
            .iter()
            .map(|&i| self.res.addr(&self.ctx, rids[i]))
            .collect::<Result<_>>()?;
        unsafe {
            let tab = self
                .slot_tab
                .contents()
                .cast::<u64>()
                .as_ptr()
                .add(slot_row * SLOT_STRIDE);
            for (u, &a) in addrs.iter().enumerate() {
                tab.add(u).write(a);
            }
            tab.add(SLOT_STRIDE - 2).write(n_res as u64);
            tab.add(SLOT_STRIDE - 1).write(union.len() as u64);
            let wm = self
                .wmap
                .contents()
                .cast::<f32>()
                .as_ptr()
                .add(slot_row * MAX_NB * SLOT_STRIDE);
            for b in 0..nb {
                let row = wm.add(b * SLOT_STRIDE);
                for (u, &i) in order.iter().enumerate() {
                    let e = union[i];
                    let w = (0..k)
                        .find(|&j| idx[b * k + j] == e)
                        .map_or(0.0, |j| wts[b * k + j]);
                    row.add(u).write(w);
                }
            }
        }
        self.event.setSignaledValue(signals.resident_ready);
        // Deadline: when the resident part is done, cut the weak misses
        // that have not landed (from the weak end, as a truncation), wait
        // for the rest.
        if deadline && !flags.is_empty() {
            self.wait_for_deadline(
                slot_row,
                signals.resident_done,
                &flags,
                &need,
                &rids,
                &order,
                n_res,
                &wmax,
            )?;
        }
        // Required misses land before lookahead starts; the deadline policy
        // may leave cut reads in flight. Deferring prefetch measured 11%
        // faster by keeping that traffic off the critical reads.
        let mut issued = 0;
        {
            let t = std::time::Instant::now();
            if let Some(h) = miss_read {
                let _ = h.join();
            }
            self.step_read_s += t.elapsed().as_secs_f64();
        }
        if let Some(next_layer) = lookahead {
            let lk = self.p.cfg.num_experts_per_tok;
            let la_idx = self.read_u32(&self.scratch.la_idx, nb * lk);
            let la_w = self.read_f32(&self.scratch.la_w, nb * lk);
            let la = union_of(&la_idx);
            self.log_lookahead(next_layer, slot_row + 1, &la, &la_idx, &la_w);
            issued = self.prefetch_async(next_layer, &la)?;
            predicted.push_back((slot_row + 1, la));
        }
        let t_fin = std::time::Instant::now();
        self.finish_miss_reads(&need, &flags, deadline)?;
        self.step_set_s += t_fin.elapsed().as_secs_f64();
        *io_s += ti.elapsed().as_secs_f64();
        self.event.setSignaledValue(signals.misses_ready);
        *turn_s += t0.elapsed().as_secs_f64();
        self.last_experts
            .push(order.iter().map(|&i| union[i]).collect());
        self.last_routes.push(idx);
        self.last_route_w.push(wts);
        self.last_miss.push(
            (0..union.len())
                .filter(|&i| missing[i])
                .map(|i| union[i])
                .collect(),
        );
        Ok((hits, total, issued))
    }
}
