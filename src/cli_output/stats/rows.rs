//! Table views built from the shared protocol types.

use cherenkov::{
    control::stats as wire,
    qwen4_exp::gpu::{self, GpuTiming},
};
use tabled::Tabled;

#[derive(Tabled)]
pub(super) struct Observation {
    #[tabled(rename = "Observed at uptime (s)", display = "decimal")]
    observed_at_uptime_seconds: f64,
    #[tabled(rename = "Time since engine load (s)", display = "decimal")]
    elapsed_seconds: f64,
    #[tabled(rename = "GPU timestamps available")]
    gpu_timestamps_available: bool,
    #[tabled(rename = "GPU timing", display = "timing_status")]
    gpu_timing: GpuTiming,
}

impl From<&wire::Observation> for Observation {
    fn from(value: &wire::Observation) -> Self {
        Self {
            observed_at_uptime_seconds: value.observed_at_uptime_seconds,
            elapsed_seconds: value.elapsed_seconds,
            gpu_timestamps_available: value.gpu_timestamps_available,
            gpu_timing: value.gpu_timing.clone(),
        }
    }
}

#[derive(Tabled)]
pub(super) struct Rates {
    #[tabled(rename = "Read bytes/s", display = "optional_decimal")]
    bytes_per_second: Option<f64>,
    #[tabled(rename = "Reads/s", display = "optional_decimal")]
    reads_per_second: Option<f64>,
    #[tabled(rename = "GPU handoff gaps (%)", display = "percent")]
    gpu_handoff_gap_fraction: Option<f64>,
    #[tabled(rename = "GPU stage coverage (%)", display = "percent")]
    gpu_stage_fraction: Option<f64>,
}

impl From<&wire::Rates> for Rates {
    fn from(value: &wire::Rates) -> Self {
        Self {
            bytes_per_second: value.bytes_per_second,
            reads_per_second: value.reads_per_second,
            gpu_handoff_gap_fraction: value.gpu_handoff_gap_fraction,
            gpu_stage_fraction: value.gpu_stage_fraction,
        }
    }
}

#[derive(Tabled)]
pub(super) struct Reads {
    #[tabled(rename = "Completed reads")]
    completed_reads: u64,
    #[tabled(rename = "Failed reads")]
    failed_reads: u64,
    #[tabled(rename = "Transferred bytes")]
    completed_bytes: u64,
    #[tabled(rename = "Summed worker read time (s)", display = "decimal")]
    read_seconds: f64,
    #[tabled(rename = "Longest read (s)", display = "decimal")]
    max_read_seconds: f64,
}

impl From<&gpu::ReadStats> for Reads {
    fn from(value: &gpu::ReadStats) -> Self {
        Self {
            completed_reads: value.completed_reads,
            failed_reads: value.failed_reads,
            completed_bytes: value.completed_bytes,
            read_seconds: value.read_seconds,
            max_read_seconds: value.max_read_seconds,
        }
    }
}

#[derive(Tabled)]
pub(super) struct Cpu {
    #[tabled(rename = "Service windows")]
    service_windows: u64,
    #[tabled(rename = "Service wall time (s)", display = "decimal")]
    service_wall_seconds: f64,
    #[tabled(rename = "Service thread CPU time (s)", display = "decimal")]
    service_cpu_seconds: f64,
    #[tabled(rename = "Prefetch wait (s)", display = "decimal")]
    prefetch_wait_seconds: f64,
    #[tabled(rename = "Demand wait (s)", display = "decimal")]
    demand_wait_seconds: f64,
}

impl From<&gpu::PhaseStats> for Cpu {
    fn from(value: &gpu::PhaseStats) -> Self {
        Self {
            service_windows: value.service_windows,
            service_wall_seconds: value.service_wall_seconds,
            service_cpu_seconds: value.service_cpu_seconds,
            prefetch_wait_seconds: value.prefetch_wait_seconds,
            demand_wait_seconds: value.demand_wait_seconds,
        }
    }
}

#[derive(Tabled)]
pub(super) struct Gpu {
    #[tabled(rename = "Valid windows")]
    gpu_windows: u64,
    #[tabled(rename = "Invalid windows")]
    invalid_gpu_windows: u64,
    #[tabled(rename = "Router to resident gap (s)", display = "decimal")]
    router_to_resident_seconds: f64,
    #[tabled(rename = "Resident stage (s)", display = "decimal")]
    resident_seconds: f64,
    #[tabled(rename = "Resident to fetched gap (s)", display = "decimal")]
    resident_to_fetched_seconds: f64,
    #[tabled(rename = "Fetched stage (s)", display = "decimal")]
    fetched_stage_seconds: f64,
}

impl From<&gpu::PhaseStats> for Gpu {
    fn from(value: &gpu::PhaseStats) -> Self {
        Self {
            gpu_windows: value.gpu_windows,
            invalid_gpu_windows: value.invalid_gpu_windows,
            router_to_resident_seconds: value.router_to_resident_seconds,
            resident_seconds: value.resident_seconds,
            resident_to_fetched_seconds: value.resident_to_fetched_seconds,
            fetched_stage_seconds: value.fetched_stage_seconds,
        }
    }
}

#[derive(Tabled)]
pub(super) struct Prediction {
    #[tabled(rename = "Target batches")]
    target_batches: u64,
    #[tabled(rename = "Predicted and selected")]
    predicted_selected: u64,
    #[tabled(rename = "Predicted but unused")]
    predicted_unused: u64,
    #[tabled(rename = "Selected without prediction")]
    selected_unpredicted: u64,
    #[tabled(rename = "Needed prefetches ready")]
    needed_prefetch_ready: u64,
    #[tabled(rename = "Needed prefetches late")]
    needed_prefetch_late: u64,
}

impl From<&gpu::PredictionStats> for Prediction {
    fn from(value: &gpu::PredictionStats) -> Self {
        Self {
            target_batches: value.target_batches,
            predicted_selected: value.predicted_selected,
            predicted_unused: value.predicted_unused,
            selected_unpredicted: value.selected_unpredicted,
            needed_prefetch_ready: value.needed_prefetch_ready,
            needed_prefetch_late: value.needed_prefetch_late,
        }
    }
}

#[derive(Tabled)]
pub(super) struct Quant {
    #[tabled(rename = "Bits")]
    bits: u8,
    #[tabled(rename = "Selections")]
    selected_experts: u64,
    #[tabled(rename = "Cut")]
    cut_experts: u64,
    #[tabled(rename = "Eligible for cut")]
    eligible_weak_misses: u64,
}

impl From<&gpu::QuantStats> for Quant {
    fn from(value: &gpu::QuantStats) -> Self {
        Self {
            bits: value.bits,
            selected_experts: value.selected_experts,
            cut_experts: value.cut_experts,
            eligible_weak_misses: value.eligible_weak_misses,
        }
    }
}

#[derive(Tabled)]
pub(super) struct Layer {
    #[tabled(rename = "Layer")]
    layer: usize,
    #[tabled(rename = "Name", display = "plain_cell")]
    name: String,
    #[tabled(inline)]
    counters: Counters,
}

impl From<&wire::Layer> for Layer {
    fn from(value: &wire::Layer) -> Self {
        Self {
            layer: value.layer,
            name: value.name.clone(),
            counters: Counters::from(&value.counters),
        }
    }
}

#[derive(Tabled)]
pub(super) struct Expert {
    #[tabled(rename = "Layer")]
    layer: usize,
    #[tabled(rename = "Expert")]
    expert: usize,
    #[tabled(inline)]
    counters: Counters,
}

impl From<&wire::Expert> for Expert {
    fn from(value: &wire::Expert) -> Self {
        Self {
            layer: value.layer,
            expert: value.expert,
            counters: Counters::from(&value.counters),
        }
    }
}

#[derive(Tabled)]
pub(super) struct Counters {
    #[tabled(rename = "Rows")]
    selected_rows: u64,
    #[tabled(rename = "Hits")]
    cache_hits: u64,
    #[tabled(rename = "Misses")]
    cache_misses: u64,
    #[tabled(rename = "Prefetch")]
    prefetch_requests: u64,
    #[tabled(rename = "Reads")]
    read_requests: u64,
    #[tabled(rename = "Requested bytes")]
    read_bytes_requested: u64,
}

impl From<&gpu::ExpertCounters> for Counters {
    fn from(value: &gpu::ExpertCounters) -> Self {
        Self {
            selected_rows: value.selected_rows,
            cache_hits: value.cache_hits,
            cache_misses: value.cache_misses,
            prefetch_requests: value.prefetch_requests,
            read_requests: value.read_requests,
            read_bytes_requested: value.read_bytes_requested,
        }
    }
}

fn decimal(value: &f64) -> String {
    format!("{value:.3}")
}

fn timing_status(value: &GpuTiming) -> String {
    match value {
        GpuTiming::NotInitialized => "not initialized".into(),
        GpuTiming::Available => "available".into(),
        GpuTiming::Unsupported => "unsupported".into(),
        GpuTiming::Disabled { reason } => format!("disabled: {}", plain_cell(reason)),
        GpuTiming::Failed { error } => format!("failed: {}", plain_cell(error)),
    }
}

fn optional_decimal(value: &Option<f64>) -> String {
    value.as_ref().map_or_else(|| "n/a".into(), decimal)
}

fn percent(value: &Option<f64>) -> String {
    optional_decimal(&value.map(|fraction| fraction * 100.0))
}

fn plain_cell(value: &str) -> String {
    // Keep external identifiers from creating table cells or terminal commands.
    value
        .chars()
        .map(|c| match c {
            '|' | '\\' | '`' | '*' | '<' | '>' => ' ',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect()
}

#[derive(Tabled)]
pub(super) struct Prefill {
    chunks: u64,
    tokens: u64,
    #[tabled(rename = "Smallest chunk (tokens)")]
    min_chunk_tokens: usize,
    #[tabled(rename = "Largest chunk (tokens)")]
    max_chunk_tokens: usize,
    #[tabled(rename = "Chunk time (s)", display = "decimal")]
    seconds: f64,
    #[tabled(rename = "Ring reuse wait (s)", display = "decimal")]
    ring_wait_seconds: f64,
}

impl From<&gpu::prefill::PrefillStats> for Prefill {
    fn from(value: &gpu::prefill::PrefillStats) -> Self {
        Self {
            chunks: value.chunks,
            tokens: value.tokens,
            min_chunk_tokens: value.min_chunk_tokens,
            max_chunk_tokens: value.max_chunk_tokens,
            seconds: value.seconds,
            ring_wait_seconds: value.ring_wait_seconds,
        }
    }
}
