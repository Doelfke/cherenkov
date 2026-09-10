use super::*;

#[test]
fn prefix_checkpoint_restores_all_hybrid_regions() {
    let Some(dir) = crate::storage::test_model_dir() else {
        return;
    };
    let packed = Packed::open(&dir).unwrap();
    for drafts in [0, 2] {
        let mut gpu = Gpu::load(
            &packed,
            64,
            &Options {
                drafts,
                pool_gb: PoolBudget::Gb(1.0),
                ..Options::default()
            },
        )
        .unwrap();
        gpu.pos = 9; // Incomplete QSA block; exercise separated q8 scale planes.
        gpu.mtp_len = if drafts == 0 { 0 } else { 9 };
        gpu.tokens = (0..9).collect();
        for (i, (b, off, n)) in gpu.prefix_regions(gpu.pos, gpu.mtp_len).iter().enumerate() {
            unsafe {
                std::ptr::write_bytes(
                    b.contents().cast::<u8>().as_ptr().add(*off),
                    (i % 251) as u8,
                    *n,
                );
            }
        }
        let checkpoint = gpu.save_prefix();
        for (b, off, n) in gpu.prefix_regions(gpu.pos, gpu.mtp_len) {
            unsafe {
                std::ptr::write_bytes(b.contents().cast::<u8>().as_ptr().add(off), 255, n);
            }
        }
        gpu.reset_request();
        gpu.restore_prefix(&checkpoint).unwrap();
        assert_eq!(gpu.pos, 9);
        assert_eq!(gpu.tokens, (0..9).collect::<Vec<u32>>());
        assert_eq!(gpu.mtp_len, checkpoint.mtp_len);
        assert_eq!(gpu.save_prefix().data, checkpoint.data);
    }
}
