use super::*;

fn memory(device_gb: usize, limit_gb: Option<usize>) -> PoolMemory {
    PoolMemory {
        device: device_gb * BYTES_PER_GB,
        fixed: 3 * BYTES_PER_GB,
        host: 2 * BYTES_PER_GB,
        allocation_limit: limit_gb.map(|n| n * BYTES_PER_GB),
        prefill: BYTES_PER_GB,
    }
}

#[test]
fn adaptive_pool_scales_without_machine_specific_clamps() {
    for (device, expected) in [(16, 5), (25, 14), (48, 37), (128, 117)] {
        assert_eq!(
            memory(device, None).bytes(PoolBudget::Adaptive).unwrap(),
            expected * BYTES_PER_GB
        );
    }
}

#[test]
fn server_budget_and_prefill_reservation_bound_every_mode() {
    let mut m = memory(48, Some(23));

    for mode in [PoolBudget::Adaptive, PoolBudget::Max, PoolBudget::Gb(19.0)] {
        assert_eq!(m.bytes(mode).unwrap(), 19 * BYTES_PER_GB);
    }

    assert!(m.bytes(PoolBudget::Gb(20.0)).is_err());

    m.prefill = 4 * BYTES_PER_GB;

    assert_eq!(m.bytes(PoolBudget::Adaptive).unwrap(), 16 * BYTES_PER_GB);
    assert!(m.bytes(PoolBudget::Gb(19.0)).is_err());
}

#[test]
fn exhausted_budgets_do_not_underflow_or_force_a_minimum() {
    for limit in [0, 2, 3, 4] {
        let m = memory(48, Some(limit));

        assert!(m.bytes(PoolBudget::Adaptive).is_err());
        assert!(m.bytes(PoolBudget::Max).is_err());
        assert!(m.bytes(PoolBudget::Gb(1.0)).is_err());
    }
}

#[test]
fn explicit_budget_is_not_clamped_to_device_recommendation() {
    assert_eq!(
        memory(48, Some(45)).bytes(PoolBudget::Gb(30.0)).unwrap(),
        30 * BYTES_PER_GB
    );
    assert_eq!(
        memory(25, None).bytes(PoolBudget::Gb(30.0)).unwrap(),
        30 * BYTES_PER_GB
    );

    for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(memory(48, None).bytes(PoolBudget::Gb(value)).is_err());
    }
}
