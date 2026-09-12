use super::*;

fn trained_pacer(ceiling: usize, samples: &[usize]) -> ChunkPacer {
    let mut pacer = ChunkPacer::new(ceiling, 4.0);

    pacer.quantum(true);

    for &tokens in samples {
        pacer.observe(tokens, 1.0);
    }

    pacer
}

fn assert_ceiling(ceiling: usize, target: f64) {
    let mut pacer = ChunkPacer::new(ceiling, target);

    assert_eq!(pacer.quantum(false), ceiling);

    let initial = if target == 0.0 {
        ceiling
    } else {
        ceiling.min(128)
    };

    assert_eq!(pacer.quantum(true), initial);

    for seconds in [0.01, 60.0, 0.01, 0.01] {
        let tokens = pacer.quantum(true);

        pacer.observe(tokens, seconds);
        assert!((ceiling.min(128)..=ceiling).contains(&pacer.tokens()));
    }
}

#[test]
fn every_mode_respects_the_configured_ceiling() {
    for ceiling in [1, 64, 128, 1024, 4096] {
        assert_ceiling(ceiling, 0.0);
        assert_ceiling(ceiling, 2.0);
    }
}

#[test]
fn contended_chunks_grow_and_shrink_by_at_most_twofold() {
    let mut pacer = ChunkPacer::new(4096, 4.0);

    assert_eq!(pacer.quantum(true), 128);

    for (tokens, seconds, expected) in [(128, 2.0, 256), (256, 0.5, 512), (512, 40.0, 256)] {
        pacer.observe(tokens, seconds);
        assert_eq!(pacer.quantum(true), expected);
    }
}

#[test]
fn damping_uses_the_previous_target_for_balanced_chunks() {
    let mut pacer = trained_pacer(4096, &[128, 256, 512]);

    assert_eq!(pacer.tokens(), 1024);

    // Balanced chunks can be slightly smaller than the requested target.
    // Damping against 990 instead of 1024 would incorrectly shrink to 495.
    pacer.observe(990, 40.0);
    assert_eq!(pacer.tokens(), 512);
}

#[test]
fn zero_target_keeps_the_configured_quantum() {
    let mut pacer = ChunkPacer::new(1024, 0.0);

    for contended in [true, false, true] {
        assert_eq!(pacer.quantum(contended), 1024);
        pacer.observe(1024, 0.1);
        assert_eq!(pacer.tokens(), 1024);
    }
}

#[test]
fn contention_restarts_small_without_exceeding_the_ceiling() {
    let mut pacer = trained_pacer(1024, &[128]);

    assert_eq!(pacer.tokens(), 256);

    assert_eq!(pacer.quantum(false), 1024);
    pacer.observe(1024, 0.1);
    assert_eq!(pacer.tokens(), 256);
    assert_eq!(pacer.quantum(true), 128);
}

#[test]
fn invalid_observations_do_not_change_the_target() {
    let mut pacer = ChunkPacer::new(1024, 2.0);

    pacer.quantum(true);

    for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        pacer.observe(128, seconds);
        assert_eq!(pacer.tokens(), 128);
    }

    pacer.observe(0, 1.0);
    assert_eq!(pacer.tokens(), 128);
}
