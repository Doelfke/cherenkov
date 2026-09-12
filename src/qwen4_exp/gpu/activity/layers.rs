//! Layer totals separate shared phase windows from precision-specific work.

use super::reads::{ReadSources, ReadStats};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct PredictionStats {
    pub target_batches: u64,
    pub predicted_selected: u64,
    pub predicted_unused: u64,
    pub selected_unpredicted: u64,
    pub predicted_resident: u64,
    pub needed_prefetch_ready: u64,
    pub needed_prefetch_late: u64,
}

impl PredictionStats {
    pub fn add(&mut self, other: Self) {
        self.target_batches += other.target_batches;
        self.predicted_selected += other.predicted_selected;
        self.predicted_unused += other.predicted_unused;
        self.selected_unpredicted += other.selected_unpredicted;
        self.predicted_resident += other.predicted_resident;
        self.needed_prefetch_ready += other.needed_prefetch_ready;
        self.needed_prefetch_late += other.needed_prefetch_late;
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct PhaseStats {
    pub service_windows: u64,
    pub service_wall_seconds: f64,
    pub service_cpu_seconds: f64,
    pub prefetch_wait_seconds: f64,
    pub demand_wait_seconds: f64,
    pub gpu_windows: u64,
    pub invalid_gpu_windows: u64,
    pub router_to_resident_seconds: f64,
    pub resident_seconds: f64,
    pub resident_to_fetched_seconds: f64,
    pub fetched_stage_seconds: f64,
    pub cpu_observation_delay_seconds: f64,
    pub cpu_prepare_seconds: f64,
    pub cpu_after_resident_release_seconds: f64,
}

impl PhaseStats {
    pub fn add(&mut self, other: Self) {
        self.service_windows += other.service_windows;
        self.service_wall_seconds += other.service_wall_seconds;
        self.service_cpu_seconds += other.service_cpu_seconds;
        self.prefetch_wait_seconds += other.prefetch_wait_seconds;
        self.demand_wait_seconds += other.demand_wait_seconds;
        self.gpu_windows += other.gpu_windows;
        self.invalid_gpu_windows += other.invalid_gpu_windows;
        self.router_to_resident_seconds += other.router_to_resident_seconds;
        self.resident_seconds += other.resident_seconds;
        self.resident_to_fetched_seconds += other.resident_to_fetched_seconds;
        self.fetched_stage_seconds += other.fetched_stage_seconds;
        self.cpu_observation_delay_seconds += other.cpu_observation_delay_seconds;
        self.cpu_prepare_seconds += other.cpu_prepare_seconds;
        self.cpu_after_resident_release_seconds += other.cpu_after_resident_release_seconds;
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct QuantStats {
    pub bits: u8,
    pub selected_rows: u64,
    pub selected_experts: u64,
    pub eligible_weak_misses: u64,
    pub cut_experts: u64,
    pub cut_batches: u64,
    pub reads: ReadSources,
}

impl QuantStats {
    fn add(&mut self, other: Self) {
        self.selected_rows += other.selected_rows;
        self.selected_experts += other.selected_experts;
        self.eligible_weak_misses += other.eligible_weak_misses;
        self.cut_experts += other.cut_experts;
        self.cut_batches += other.cut_batches;

        self.reads.add(other.reads);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub prediction: PredictionStats,
    pub phases: PhaseStats,
    pub quant: [QuantStats; 3],
}

impl Default for LayerStats {
    fn default() -> Self {
        Self {
            prediction: PredictionStats::default(),
            phases: PhaseStats::default(),
            quant: [4, 3, 2].map(|bits| QuantStats {
                bits,
                ..QuantStats::default()
            }),
        }
    }
}

impl LayerStats {
    pub fn add(&mut self, other: &Self) {
        self.prediction.add(other.prediction);
        self.phases.add(other.phases);

        for (a, &b) in self.quant.iter_mut().zip(&other.quant) {
            a.add(b);
        }
    }

    pub fn reads(&self) -> ReadStats {
        let mut total = ReadStats::default();

        for q in &self.quant {
            total.add(q.reads.total());
        }

        total
    }
}
