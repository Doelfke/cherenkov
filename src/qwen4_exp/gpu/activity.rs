//! Cumulative routed-expert activity. Verification and draft rows count even
//! when later rolled back. IO counters describe requested reads, not completion.

pub(super) mod layers;
pub(super) mod reads;
use super::*;
pub use layers::LayerStats;
use reads::ReadSource;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct ExpertCounters {
    pub selected_rows: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefetch_requests: u64,
    pub read_requests: u64,
    pub read_bytes_requested: u64,
}

impl ExpertCounters {
    pub fn add(&mut self, other: Self) {
        self.selected_rows += other.selected_rows;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.prefetch_requests += other.prefetch_requests;
        self.read_requests += other.read_requests;
        self.read_bytes_requested += other.read_bytes_requested;
    }

    fn lookup(&mut self, rows: usize, resident: bool) {
        self.selected_rows += rows as u64;
        self.cache_hits += u64::from(resident);
        self.cache_misses += u64::from(!resident);
    }
}

#[derive(Debug, Default, Clone)]
pub struct ExpertActivity {
    pub layer_prefixes: Vec<String>,
    pub experts_per_layer: usize,
    pub records: Vec<ExpertCounters>,
    pub layers: Vec<LayerStats>,
    pub elapsed_seconds: f64,
    pub gpu_timestamps_available: bool,
    pub gpu_timing: GpuTiming,
}

impl ExpertActivity {
    pub(super) fn layer_for_record(&self, record: usize) -> usize {
        record / self.experts_per_layer
    }

    pub(crate) fn validate_dimensions(&self) -> Result<()> {
        anyhow::ensure!(
            self.layer_prefixes.len() == self.layers.len(),
            "expert activity layer names and stats differ in length"
        );
        anyhow::ensure!(
            self.layers.len().checked_mul(self.experts_per_layer) == Some(self.records.len()),
            "expert activity record count does not match layer dimensions"
        );

        Ok(())
    }

    pub(super) fn new(layout: &crate::qwen4_exp::ExpertLayout) -> Self {
        Self {
            layers: vec![LayerStats::default(); layout.layers],
            elapsed_seconds: 0.0,
            gpu_timestamps_available: false,
            gpu_timing: GpuTiming::NotInitialized,
            layer_prefixes: layout.layer_prefixes.clone(),
            experts_per_layer: layout.experts,
            records: vec![ExpertCounters::default(); layout.layers * layout.experts],
        }
    }

    pub(super) fn lookup(&mut self, record: usize, rows: usize, resident: bool) {
        self.records[record].lookup(rows, resident);
    }

    pub(super) fn read(&mut self, record: usize, bytes: usize) {
        self.records[record].read_requests += 1;
        self.records[record].read_bytes_requested += bytes as u64;
    }
}

impl Gpu<'_> {
    pub(super) fn record_cut_eligible(
        &mut self,
        layer: usize,
        records: &[usize],
        missing: &[bool],
        weights: &[f32],
    ) {
        if self.cut_w <= 0.0 || self.fake_experts {
            return;
        }

        for ((&record, &missing), &weight) in records.iter().zip(missing).zip(weights) {
            if missing && weight < self.cut_w {
                let kind = self.res.kind(record) as usize;
                self.activity.layers[layer].quant[kind].eligible_weak_misses += 1;
            }
        }
    }
    pub fn expert_activity(&self) -> &ExpertActivity {
        &self.activity
    }

    pub(super) fn record_routed_experts(&mut self, layer: usize, ids: &[u32], experts: &[u32]) {
        for &expert in experts {
            let record = self.record_id(layer, expert);
            let rows = ids.iter().filter(|&&id| id == expert).count();

            self.activity
                .lookup(record, rows, self.res.is_member(record));
        }
    }

    pub(super) fn record_read_plan(
        &mut self,
        plan: &mut residency::ReadPlan,
        need: &[usize],
        source: ReadSource,
    ) {
        if self.fake_experts {
            return;
        }

        if let Some(&record) = need.first() {
            plan.observe(
                &self.read_tracker,
                self.activity.layer_for_record(record),
                source,
            );
        }

        for (index, bytes) in plan.read_requests() {
            self.activity.read(need[index], bytes);
        }
    }

    pub fn copy_expert_activity(&self, snapshot: &mut ExpertActivity) {
        snapshot
            .layer_prefixes
            .clone_from(&self.activity.layer_prefixes);
        snapshot.records.clone_from(&self.activity.records);
        snapshot.layers.clone_from(&self.activity.layers);

        snapshot.experts_per_layer = self.activity.experts_per_layer;
        snapshot.elapsed_seconds = self.activity_started.elapsed().as_secs_f64();
        snapshot.gpu_timestamps_available = self.phase_timer.is_some();

        snapshot.gpu_timing.clone_from(&self.activity.gpu_timing);

        for (layer, reads) in snapshot.layers.iter_mut().zip(self.read_tracker.snapshot()) {
            for (quant, reads) in layer.quant.iter_mut().zip(reads) {
                quant.reads = reads;
            }
        }
    }

    pub(super) fn record_quant_selection(&mut self, layer: usize, ids: &[u32], experts: &[u32]) {
        for &expert in experts {
            let kind = self.res.kind(self.record_id(layer, expert));
            let stats = &mut self.activity.layers[layer].quant[kind as usize];
            stats.selected_experts += 1;
            stats.selected_rows += ids.iter().filter(|&&id| id == expert).count() as u64;
        }
    }

    pub(super) fn record_prediction(&mut self, layer: usize, predicted: &[u32], selected: &[u32]) {
        let stats = &mut self.activity.layers[layer].prediction;
        stats.target_batches += 1;
        stats.selected_unpredicted +=
            selected.iter().filter(|id| !predicted.contains(id)).count() as u64;

        for expert in predicted {
            if !selected.contains(expert) {
                stats.predicted_unused += 1;

                continue;
            }

            stats.predicted_selected += 1;
            let record = layer * self.activity.experts_per_layer + *expert as usize;
            let ticket = self
                .pending
                .as_ref()
                .and_then(|pending| pending.tickets.iter().find(|(r, _)| *r == record));

            if let Some((_, ticket)) = ticket {
                let ready = ticket.done();
                stats.needed_prefetch_ready += u64::from(ready);
                stats.needed_prefetch_late += u64::from(!ready);
            }
        }
    }
}
