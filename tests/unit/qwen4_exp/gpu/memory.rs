use super::*;

#[test]
fn memory_snapshot_tracks_context_and_counts_weight_mapping_once() {
    let Some(dir) = crate::storage::test_model_dir() else {
        return;
    };
    let packed = Packed::open(&dir).unwrap();
    let mut previous = None;

    for context in [64, 128] {
        let gpu = Gpu::load(
            &packed,
            context,
            &Options {
                drafts: 0,
                pool_gb: PoolBudget::Gb(0.25),
                ..Options::default()
            },
        )
        .unwrap();
        let stats = gpu.memory_stats();

        assert_eq!(stats.context_capacity_tokens, context);
        assert!(!stats.mtp_enabled);
        assert_eq!(
            stats.mapped_weight_buffer_bytes,
            packed.dense.len().next_multiple_of(crate::metal::PAGE_SIZE)
        );
        assert!(stats.kv_index_capacity_bytes > 0);
        assert!(stats.metal_allocated_bytes_observed >= stats.kv_index_capacity_bytes as u64);
        assert!(stats.resident_experts <= stats.expert_pool_slots);

        if let Some(prior) = previous {
            assert!(stats.kv_index_capacity_bytes > prior);
        }

        previous = Some(stats.kv_index_capacity_bytes);
    }
}

#[test]
fn expert_activity_counts_fetches_once_and_survives_sequence_reset() {
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
    let k = packed.cfg.num_experts_per_tok;
    let ids: Vec<u32> = (0..k as u32).collect();
    let weights = vec![1.0_f32 / k as f32; k];

    unsafe {
        std::ptr::copy_nonoverlapping(
            ids.as_ptr(),
            gpu.scratch.topk_idx.contents().cast::<u32>().as_ptr(),
            k,
        );
        std::ptr::copy_nonoverlapping(
            weights.as_ptr(),
            gpu.scratch.topk_w.contents().cast::<f32>().as_ptr(),
            k,
        );
    }

    for seq in [10, 20] {
        gpu.step_no += 1;

        gpu.event.setSignaledValue(seq);
        gpu.service_block(
            0,
            0,
            1,
            seq,
            &mut Default::default(),
            None,
            &mut 0.0,
            &mut 0.0,
        )
        .unwrap();
    }

    gpu.reset_request();

    let records = &gpu.expert_activity().records;

    for counters in &records[..k] {
        assert_eq!(counters.selected_rows, 2);
        assert_eq!(counters.cache_misses, 1);
        assert_eq!(counters.cache_hits, 1);
        assert_eq!(counters.read_requests, 1);
        assert_eq!(
            counters.read_bytes_requested,
            packed.manifest.experts.record_stride
        );
    }

    assert!(
        records[k..]
            .iter()
            .all(|c| c.selected_rows == 0 && c.read_requests == 0)
    );
}
