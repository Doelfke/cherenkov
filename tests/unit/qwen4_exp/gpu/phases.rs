use super::*;

#[test]
fn phase_totals_partition_windows_and_reject_invalid_timestamps() {
    let mut stats = super::super::activity::layers::PhaseStats::default();

    assert!(accumulate_phases(&mut stats, &[10, 12, 17, 20, 30], 0.001));
    assert_eq!(stats.gpu_windows, 1);
    assert_eq!(stats.router_to_resident_seconds, 0.002);
    assert_eq!(stats.resident_seconds, 0.005);
    assert_eq!(stats.resident_to_fetched_seconds, 0.003);
    assert_eq!(stats.fetched_stage_seconds, 0.01);
    assert!(!accumulate_phases(&mut stats, &[10, 0, 17, 20, 30], 1.0));
    assert!(!accumulate_phases(&mut stats, &[10, 12, 11, 20, 30], 1.0));
    assert_eq!(stats.gpu_windows, 1);
}

fn encode_probe(
    ctx: &MetalContext,
    timer: &PhaseTimer,
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    start: Option<usize>,
    end: usize,
) {
    let lib = ctx
        .compile_library(crate::kernels::CLOCK_PROBE_MSL)
        .unwrap();
    let pipeline = ctx.pipeline(&lib, "clock_probe").unwrap();
    let buffer = ctx.new_buffer(4).unwrap();
    let enc = timer.encoder(cb, start, end).unwrap();

    enc.setComputePipelineState(&pipeline);

    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buffer), 0, 0);
        enc.setBytes_length_atIndex(NonNull::from(&32u32).cast(), 4, 1);
    }

    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
}

#[test]
fn metal_phase_samples_expose_cpu_release_gaps() {
    let ctx = MetalContext::new().unwrap();
    let Some(timer) = PhaseTimer::new(&ctx, 1).unwrap() else {
        return;
    };
    let event = ctx.device.newSharedEvent().unwrap();
    let event_ref = ProtocolObject::from_ref(&*event);
    let before = sample_clock(&ctx);
    let cb = ctx.queue.commandBuffer().unwrap();

    encode_probe(&ctx, &timer, &cb, None, 0);
    cb.encodeSignalEvent_value(event_ref, 1);
    cb.encodeWaitForEvent_value(event_ref, 2);
    encode_probe(&ctx, &timer, &cb, Some(1), 2);
    cb.encodeSignalEvent_value(event_ref, 3);
    cb.encodeWaitForEvent_value(event_ref, 4);
    encode_probe(&ctx, &timer, &cb, Some(3), 4);

    let wall_started = std::time::Instant::now();

    cb.commit();
    assert!(event.waitUntilSignaledValue_timeoutMS(1, 5000));
    std::thread::sleep(std::time::Duration::from_millis(10));
    event.setSignaledValue(2);
    assert!(event.waitUntilSignaledValue_timeoutMS(3, 5000));
    std::thread::sleep(std::time::Duration::from_millis(15));
    event.setSignaledValue(4);
    cb.waitUntilCompleted();

    let wall_seconds = wall_started.elapsed().as_secs_f64();

    let scale = clock_scale(before, sample_clock(&ctx)).unwrap();
    let mut stats = super::super::activity::layers::PhaseStats::default();

    assert!(accumulate_phases(
        &mut stats,
        &timer.timestamps().unwrap(),
        scale
    ));
    assert!(stats.router_to_resident_seconds >= 0.008, "{stats:?}");
    assert!(stats.resident_to_fetched_seconds >= 0.012, "{stats:?}");
    assert!(stats.resident_seconds > 0.0);

    let measured = stats.router_to_resident_seconds
        + stats.resident_seconds
        + stats.resident_to_fetched_seconds
        + stats.fetched_stage_seconds;

    assert!(
        measured <= wall_seconds * 1.05,
        "GPU phases {measured}s exceed wall time {wall_seconds}s"
    );
}

#[test]
fn clock_calibration_uses_nanoseconds_independently_of_mach_timebase() {
    let before = ClockSample {
        cpu_ns: 10_000_000_000,
        gpu: 20_000_000_000,
    };
    let after = ClockSample {
        cpu_ns: 10_025_000_000,
        gpu: 20_050_000_000,
    };
    let scale = clock_scale(before, after).unwrap();
    let mut stats = super::super::activity::layers::PhaseStats::default();

    assert!(accumulate_phases(
        &mut stats,
        &[
            20_000_000_000,
            20_020_000_000,
            20_022_000_000,
            20_048_000_000,
            20_050_000_000
        ],
        scale
    ));
    assert!((stats.router_to_resident_seconds - 0.010).abs() < 1e-12);
    assert!((stats.resident_to_fetched_seconds - 0.013).abs() < 1e-12);
    assert!(clock_scale(before, before).is_none());
    assert!(clock_scale(after, before).is_none());
}

#[test]
fn cpu_observation_delay_converts_mach_ticks_before_subtracting() {
    for timebase in [
        MachTimebase {
            numerator: 125,
            denominator: 3,
        },
        MachTimebase {
            numerator: 1,
            denominator: 1,
        },
    ] {
        let ticks = 240_000_000_000;
        let before_ns = timebase.nanoseconds(ticks);
        let delay_ticks =
            12_000_000 * u64::from(timebase.denominator) / u64::from(timebase.numerator);
        let observed = timebase.nanoseconds(ticks + delay_ticks);
        let before = ClockSample {
            cpu_ns: before_ns,
            gpu: 100_000_000,
        };
        let delay = observation_delay(observed, 110_000_000, before, SECONDS_PER_NANOSECOND);

        assert!((delay - 0.002).abs() < 1e-12, "{delay}");
    }
}

#[test]
fn timer_failure_and_unsupported_hardware_have_distinct_statuses() {
    let (timer, status) = PhaseTimer::initialization_result(Ok(None));

    assert!(timer.is_none());
    assert!(matches!(status, GpuTiming::Unsupported));

    let (timer, status) =
        PhaseTimer::initialization_result(Err(anyhow::anyhow!("allocation failed")));

    assert!(timer.is_none());
    assert!(matches!(status, GpuTiming::Failed { error } if error == "allocation failed"));
}

#[test]
fn expert_service_exports_cpu_gpu_and_read_stats_from_one_window() {
    let Some(dir) = crate::storage::test_model_dir() else {
        return;
    };
    let packed = Packed::open(&dir).unwrap();
    let mut gpu = Gpu::load(
        &packed,
        64,
        &Options {
            drafts: 0,
            pool_gb: PoolBudget::Gb(0.25),
            ..Options::default()
        },
    )
    .unwrap();
    let Some(timer) = &gpu.phase_timer else {
        return;
    };
    let k = packed.cfg.num_experts_per_tok;

    unsafe {
        let ids = gpu.scratch.topk_idx.contents().cast::<u32>().as_ptr();
        let weights = gpu.scratch.topk_w.contents().cast::<f32>().as_ptr();

        for i in 0..k {
            ids.add(i).write(i as u32);
            weights.add(i).write(1.0 / k as f32);
        }
    }

    let wall_started = std::time::Instant::now();
    let before = gpu.phase_clock();
    let event = ProtocolObject::from_ref(&*gpu.event);
    let resident = ProtocolObject::from_ref(&*gpu.event_res);
    let cb = gpu.ctx.queue.commandBuffer().unwrap();

    encode_probe(&gpu.ctx, timer, &cb, None, 0);
    cb.encodeSignalEvent_value(event, 1);
    cb.encodeWaitForEvent_value(event, 2);
    encode_probe(&gpu.ctx, timer, &cb, Some(1), 2);
    cb.encodeSignalEvent_value(resident, 3);
    cb.encodeWaitForEvent_value(event, 4);
    encode_probe(&gpu.ctx, timer, &cb, Some(3), 4);
    cb.commit();

    gpu.step_no = 1;

    gpu.service_block(
        0,
        0,
        1,
        1,
        &mut Default::default(),
        None,
        &mut 0.0,
        &mut 0.0,
    )
    .unwrap();
    cb.waitUntilCompleted();
    gpu.collect_phases(&[(0, 0)], before);

    let mut snapshot = ExpertActivity::default();

    gpu.copy_expert_activity(&mut snapshot);

    assert_service_window(
        snapshot.layers[0].phases,
        wall_started.elapsed().as_secs_f64(),
    );
    assert!(snapshot.gpu_timestamps_available);
    assert!(matches!(snapshot.gpu_timing, GpuTiming::Available));

    let quant = snapshot.layers[0].quant[0];

    assert_eq!(quant.reads.demand.completed_reads, k as u64);
    assert_eq!(quant.selected_experts, k as u64);
}

fn assert_service_window(phases: super::super::activity::layers::PhaseStats, wall_seconds: f64) {
    assert_eq!(phases.gpu_windows, 1, "{phases:?}");
    assert_eq!(phases.invalid_gpu_windows, 0);
    assert_eq!(phases.service_windows, 1);
    assert!(phases.service_cpu_seconds > 0.0);
    assert!(phases.cpu_prepare_seconds > 0.0);
    assert!(phases.cpu_observation_delay_seconds > 0.0, "{phases:?}");
    assert!(
        phases.cpu_observation_delay_seconds < wall_seconds,
        "{phases:?}"
    );
    assert!(phases.service_wall_seconds >= phases.service_cpu_seconds);
}
