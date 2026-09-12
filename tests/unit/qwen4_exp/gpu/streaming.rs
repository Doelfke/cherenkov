use super::*;

fn options() -> Options {
    Options {
        drafts: 0,
        pool_gb: PoolBudget::Gb(0.25),
        ..Options::default()
    }
}

#[test]
fn prediction_counts_readiness_before_joining_pending_reads() {
    let Some(dir) = crate::storage::test_model_dir() else {
        return;
    };
    let packed = Packed::open(&dir).unwrap();
    let mut gpu = Gpu::load(&packed, 64, &options()).unwrap();
    let ready = gpu.read_tracker.ticket(0, 0, ReadSource::Prefetch, 32);

    ready.measure(|| 32);

    let late = gpu.read_tracker.ticket(0, 0, ReadSource::Prefetch, 32);
    gpu.pending = Some(PendingRead {
        thread: None,
        records: vec![0, 1],
        tickets: vec![(0, ready), (1, late)],
    });

    gpu.record_prediction(0, &[0, 1, 2], &[0, 1, 3]);

    let stats = gpu.activity.layers[0].prediction;

    assert_eq!(stats.predicted_selected, 2);
    assert_eq!(stats.predicted_unused, 1);
    assert_eq!(stats.selected_unpredicted, 1);
    assert_eq!(stats.needed_prefetch_ready, 1);
    assert_eq!(stats.needed_prefetch_late, 1);
    gpu.pending.take();
}

#[test]
fn deadline_cuts_count_actual_precision_and_only_eligible_weak_misses() {
    let Some(dir) = crate::storage::test_model_dir() else {
        return;
    };
    let packed = Packed::open(&dir).unwrap();
    let mut gpu = Gpu::load(&packed, 64, &options()).unwrap();
    let records = [0, 1, 2];

    // Only metadata is exercised here; no low-bit bytes reach a kernel.
    gpu.res
        .set_low_bit_store(tempfile::tempfile().unwrap(), 64, 0);
    gpu.res.acquire(&gpu.ctx, &records, 1).unwrap();
    gpu.res.set_kind(0, 2);
    gpu.res.set_kind(1, 1);

    let weights = [0.01, 0.02, 0.2];

    gpu.record_cut_eligible(0, &records, &[true; 3], &weights);
    assert_eq!(gpu.activity.layers[0].quant[2].eligible_weak_misses, 0);

    gpu.cut_w = 0.08;

    gpu.record_cut_eligible(0, &records, &[true; 3], &weights);

    let flags = [
        residency::Landed::new(false),
        residency::Landed::new(false),
        residency::Landed::new(true),
    ];

    gpu.event_res.setSignaledValue(1);
    gpu.wait_for_deadline(0, 1, &flags, &records, &records, &[2, 1, 0], 0, &weights)
        .unwrap();

    let quant = gpu.activity.layers[0].quant;

    for q in &quant[1..] {
        assert_eq!(q.eligible_weak_misses, 1);
        assert_eq!(q.cut_experts, 1);
        assert_eq!(q.cut_batches, 1);
    }

    assert_eq!(quant[0].cut_experts, 0);
    assert_eq!(gpu.step_cut, 2);
    // These flags model intentionally unfinished reads; there are no IO writers.
    gpu.inflight.clear();
}
