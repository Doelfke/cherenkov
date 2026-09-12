//! Timestamp the existing compute passes; resolve only after their normal wait.

use super::*;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLComputePassDescriptor, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
    MTLStorageMode,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

const SECONDS_PER_NANOSECOND: f64 = 1e-9;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GpuTiming {
    #[default]
    NotInitialized,
    Available,
    Unsupported,
    Disabled {
        reason: String,
    },
    Failed {
        error: String,
    },
}

#[derive(Clone, Copy)]
pub(super) struct ClockSample {
    cpu_ns: u64,
    gpu: u64,
}

/// CPU markers use the same nanosecond timebase as Metal's CPU samples.
#[derive(Default, Clone, Copy)]
struct CpuWindow {
    observed: u64,
    resident: u64,
    fetched: u64,
}

pub(super) enum CpuPhase {
    Observed,
    ResidentRelease,
    FetchedRelease,
}

pub(super) struct PhaseTimer {
    buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    // Router end, resident start/end, fetched start. The next router end
    // closes this window and opens the next; the final sample closes the head.
    count: usize,
    cpu: RefCell<Vec<CpuWindow>>,
    timebase: MachTimebase,
}

impl PhaseTimer {
    pub(super) fn initialize(
        ctx: &MetalContext,
        slots: usize,
        enabled: bool,
    ) -> (Option<Self>, GpuTiming) {
        if !enabled {
            return (
                None,
                GpuTiming::Disabled {
                    reason: "diagnostic layer cap changes the compute-pass layout".into(),
                },
            );
        }

        Self::initialization_result(Self::new(ctx, slots))
    }

    fn initialization_result(result: Result<Option<Self>>) -> (Option<Self>, GpuTiming) {
        match result {
            Ok(Some(timer)) => (Some(timer), GpuTiming::Available),
            Ok(None) => (None, GpuTiming::Unsupported),
            Err(error) => {
                let error = format!("{error:#}");

                eprintln!("GPU phase timing unavailable: {error}");

                (None, GpuTiming::Failed { error })
            }
        }
    }

    pub(super) fn new(ctx: &MetalContext, slots: usize) -> Result<Option<Self>> {
        if !ctx
            .device
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Ok(None);
        }

        let sets = ctx
            .device
            .counterSets()
            .context("Metal counter sets unavailable")?;
        let set = sets
            .iter()
            .find(|set| &*set.name() == unsafe { MTLCommonCounterSetTimestamp })
            .context("Metal timestamp counter unavailable")?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();

        descriptor.setCounterSet(Some(&set));
        descriptor.setStorageMode(MTLStorageMode::Shared);

        let count = 4 * slots + 1;

        unsafe {
            descriptor.setSampleCount(count);
        }

        let buffer = ctx
            .device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|e| anyhow::anyhow!("timestamp buffer: {e}"))?;

        Ok(Some(Self {
            buffer,
            count,
            cpu: RefCell::new(vec![CpuWindow::default(); slots]),
            timebase: MachTimebase::new()?,
        }))
    }

    fn encoder(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        start: Option<usize>,
        end: usize,
    ) -> Result<Retained<Enc>> {
        ensure_indices(self.count, start, end)?;

        let descriptor = MTLComputePassDescriptor::new();
        let attachment = unsafe {
            descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0)
        };

        attachment.setSampleBuffer(Some(&self.buffer));

        unsafe {
            attachment.setStartOfEncoderSampleIndex(start.unwrap_or(usize::MAX));
            attachment.setEndOfEncoderSampleIndex(end);
        }

        cb.computeCommandEncoderWithDescriptor(&descriptor)
            .context("timed compute encoder")
    }

    fn timestamps(&self) -> Option<Vec<u64>> {
        let data = unsafe { self.buffer.resolveCounterRange(NSRange::new(0, self.count)) }?;
        let bytes = unsafe { data.as_bytes_unchecked() };

        if bytes.len() != self.count * 8 {
            return None;
        }

        Some(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_ne_bytes(*b))
                .collect(),
        )
    }
}

fn ensure_indices(count: usize, start: Option<usize>, end: usize) -> Result<()> {
    anyhow::ensure!(
        end < count && start.is_none_or(|s| s < count),
        "timestamp index exceeds buffer"
    );

    Ok(())
}

fn sample_clock(ctx: &MetalContext) -> ClockSample {
    let (mut cpu, mut gpu) = (0, 0);

    unsafe {
        ctx.device
            .sampleTimestamps_gpuTimestamp((&mut cpu).into(), (&mut gpu).into());
    }

    ClockSample { cpu_ns: cpu, gpu }
}

struct MachTimebase {
    numerator: u32,
    denominator: u32,
}

impl MachTimebase {
    // libc deprecated its Mach bindings, not the underlying macOS API.
    #[allow(deprecated)]
    fn new() -> Result<Self> {
        let mut timebase = libc::mach_timebase_info_data_t { numer: 0, denom: 0 };
        let status = unsafe { libc::mach_timebase_info(&mut timebase) };

        anyhow::ensure!(
            status == 0 && timebase.denom != 0,
            "Mach timebase unavailable"
        );

        Ok(Self {
            numerator: timebase.numer,
            denominator: timebase.denom,
        })
    }

    fn nanoseconds(&self, ticks: u64) -> u64 {
        // Widen before multiplication: absolute uptimes can exceed u64 / numer.
        (u128::from(ticks) * u128::from(self.numerator) / u128::from(self.denominator)) as u64
    }
}

pub(super) fn thread_cpu_seconds() -> f64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };

    if status != 0 {
        return 0.0;
    }

    time.tv_sec as f64 + time.tv_nsec as f64 / 1e9
}

impl Gpu<'_> {
    pub(super) fn phase_encoder(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        start: Option<usize>,
        end: usize,
    ) -> Result<Retained<Enc>> {
        match &self.phase_timer {
            Some(timer) => timer.encoder(cb, start, end),
            None => cb.computeCommandEncoder().context("encoder"),
        }
    }

    pub(super) fn phase_clock(&self) -> Option<ClockSample> {
        self.phase_timer.as_ref().map(|_| sample_clock(&self.ctx))
    }

    #[allow(deprecated)]
    pub(super) fn phase_cpu_mark(&self, slot: usize, phase: CpuPhase) {
        let Some(timer) = &self.phase_timer else {
            return;
        };
        let now = timer
            .timebase
            .nanoseconds(unsafe { libc::mach_absolute_time() });
        let mut windows = timer.cpu.borrow_mut();

        match phase {
            CpuPhase::Observed => windows[slot].observed = now,
            CpuPhase::ResidentRelease => windows[slot].resident = now,
            CpuPhase::FetchedRelease => windows[slot].fetched = now,
        }
    }

    pub(super) fn collect_trunk_phases(
        &mut self,
        layers: usize,
        mtp: Option<usize>,
        clock: Option<ClockSample>,
    ) {
        let mut slots: Vec<_> = self.layers[..layers]
            .iter()
            .enumerate()
            .map(|(slot, layer)| (slot, layer.moe.record_layer))
            .collect();

        if let Some(record) = mtp {
            slots.push((self.layers.len(), record));
        }

        self.collect_phases(&slots, clock);
    }

    pub(super) fn collect_phases(&mut self, slots: &[(usize, usize)], before: Option<ClockSample>) {
        let Some(timer) = &self.phase_timer else {
            return;
        };
        let Some(before) = before else {
            return;
        };
        let after = sample_clock(&self.ctx);
        let timestamps = timer.timestamps();
        let scale = clock_scale(before, after);

        for &(slot, layer) in slots {
            let stats = &mut self.activity.layers[layer].phases;
            let points = timestamps
                .as_ref()
                .and_then(|v| v.get(4 * slot..4 * slot + 5))
                .filter(|p| p[0] >= before.gpu && p[4] <= after.gpu);
            let valid = points
                .zip(scale)
                .is_some_and(|(p, scale)| accumulate_phases(stats, p, scale));

            if !valid {
                stats.invalid_gpu_windows += 1;
            }

            let cpu = timer.cpu.borrow()[slot];
            stats.cpu_prepare_seconds +=
                cpu.resident.saturating_sub(cpu.observed) as f64 * SECONDS_PER_NANOSECOND;
            stats.cpu_after_resident_release_seconds +=
                cpu.fetched.saturating_sub(cpu.resident) as f64 * SECONDS_PER_NANOSECOND;

            if let Some((p, scale)) = points.zip(scale).filter(|_| valid) {
                stats.cpu_observation_delay_seconds +=
                    observation_delay(cpu.observed, p[0], before, scale);
            }
        }
    }
}

fn observation_delay(observed_ns: u64, gpu_start: u64, before: ClockSample, scale: f64) -> f64 {
    let gpu_offset = signed_delta(gpu_start, before.gpu) * scale;
    let cpu_offset = signed_delta(observed_ns, before.cpu_ns) * SECONDS_PER_NANOSECOND;

    (cpu_offset - gpu_offset).max(0.0)
}

fn signed_delta(a: u64, b: u64) -> f64 {
    if a >= b {
        return (a - b) as f64;
    }

    -((b - a) as f64)
}

fn clock_scale(before: ClockSample, after: ClockSample) -> Option<f64> {
    let cpu = after.cpu_ns.checked_sub(before.cpu_ns)?;
    let gpu = after.gpu.checked_sub(before.gpu)?;

    if cpu == 0 || gpu == 0 {
        return None;
    }

    // Metal's calibrated CPU timestamps are already nanoseconds, not Mach ticks.
    Some(cpu as f64 / gpu as f64 * SECONDS_PER_NANOSECOND)
}

fn accumulate_phases(
    stats: &mut super::activity::layers::PhaseStats,
    p: &[u64],
    scale: f64,
) -> bool {
    if p.iter().any(|&t| t == 0 || t == u64::MAX) || p.windows(2).any(|p| p[1] < p[0]) {
        return false;
    }

    stats.gpu_windows += 1;
    stats.router_to_resident_seconds += (p[1] - p[0]) as f64 * scale;
    stats.resident_seconds += (p[2] - p[1]) as f64 * scale;
    stats.resident_to_fetched_seconds += (p[3] - p[2]) as f64 * scale;
    stats.fetched_stage_seconds += (p[4] - p[3]) as f64 * scale;

    true
}

#[cfg(test)]
#[path = "../../../tests/unit/qwen4_exp/gpu/phases.rs"]
mod tests;
