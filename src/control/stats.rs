//! Statistics responses shared by the server and its clients.

use crate::qwen4_exp::gpu::{ExpertCounters, GpuTiming, LayerStats, ReadStats};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot<T> {
    #[serde(flatten)]
    pub observation: Observation,
    #[serde(flatten)]
    pub data: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub observed_at_uptime_seconds: f64,
    pub elapsed_seconds: f64,
    pub gpu_timestamps_available: bool,
    pub gpu_timing: GpuTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub streaming: LayerStats,
    pub reads: ReadStats,
    #[serde(flatten)]
    pub rates: Rates,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rates {
    pub bytes_per_second: Option<f64>,
    pub reads_per_second: Option<f64>,
    pub gpu_handoff_gap_fraction: Option<f64>,
    pub gpu_stage_fraction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page<T> {
    pub offset: usize,
    pub next_offset: Option<usize>,
    pub total: usize,
    pub data: Vec<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub layer: usize,
    pub name: String,
    pub experts: usize,
    #[serde(flatten)]
    pub counters: ExpertCounters,
    pub streaming: LayerStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    pub layer: usize,
    pub expert: usize,
    #[serde(flatten)]
    pub counters: ExpertCounters,
}
