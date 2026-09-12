use super::*;

#[test]
fn matching_checks_mtp_lookahead_and_exact_prompt_state() {
    assert!(matches(&[1, 2], Some(3), false, &[1, 2, 3, 4]));
    assert!(!matches(&[1, 2], Some(3), false, &[1, 2, 9]));
    assert!(!matches(&[1, 2], Some(3), false, &[1, 2]));
    assert!(matches(&[1, 2], Some(3), true, &[1, 2]));
    assert!(matches(&[1, 2], None, false, &[1, 2, 9]));
    assert!(!matches(&[1, 2], None, true, &[1, 9, 2]));
    assert!(!matches(&[1, 2], None, true, &[1]));
}

#[test]
fn chunks_respect_quantum_capacity_and_prompt_boundaries() {
    for (remaining, capacity, quantum, expected) in [
        (275, 4096, 128, 92),
        (25740, 4096, 128, 128),
        (25740, 4096, 1024, 990),
        (25740, 4096, 4096, 3678),
        (275, 64, 1024, 55),
        (17, 4096, 4096, 17),
    ] {
        assert_eq!(chunk_len(remaining, capacity, quantum).unwrap(), expected);
    }

    assert!(chunk_len(10, 0, 128).is_err());
    assert!(chunk_len(10, 128, 0).is_err());
    assert!(chunk_len(0, 128, 128).is_err());
}

#[test]
fn pacing_ignores_tails_memory_limits_and_the_small_row_path() {
    for (remaining, capacity, engine, eligible) in [
        (4096, 4096, true, true),
        (25740, 4096, true, true),
        (4, 4096, true, false),
        (1000, 4096, true, false),
        (4096, 512, true, false),
        (4096, 4096, false, false),
    ] {
        assert_eq!(pacing_sample(remaining, capacity, 1024, engine), eligible);
    }
}
