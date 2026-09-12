use super::history::History;
use cherenkov::control::stats::{Snapshot, Summary};

fn observation(time: f64, bytes: u64) -> Snapshot<Summary> {
    let mut snapshot: Snapshot<Summary> =
        serde_json::from_value(crate::cli_output::tests::summary()).unwrap();
    snapshot.observation.observed_at_uptime_seconds = time;
    snapshot.observation.elapsed_seconds = time - 1.0;
    snapshot.data.reads.completed_bytes = bytes;

    snapshot
}

fn recorded(samples: &[(f64, u64)]) -> History {
    let mut history = History::default();

    for &(time, bytes) in samples {
        history.observe(&observation(time, bytes));
    }

    history
}

#[test]
fn interval_rates_use_actual_time_and_integer_byte_deltas() {
    let mut history = History::default();
    let first = observation(10.0, u64::MAX - 8_000_000);
    let mut second = observation(12.0, u64::MAX - 2_000_000);
    second.data.streaming.phases.demand_wait_seconds = 0.4;

    history.observe(&first);
    history.observe(&second);

    let values = history.samples.back().unwrap().values;

    assert_eq!(values, [Some(3.0), Some(200.0), None]);
}

#[test]
fn gpu_coverage_uses_measured_windows_and_idle_is_missing() {
    let mut history = History::default();
    let mut first = observation(10.0, 0);
    first.observation.gpu_timestamps_available = true;
    let mut second = first.clone();
    second.observation.observed_at_uptime_seconds += 1.0;
    second.observation.elapsed_seconds += 1.0;
    second.data.streaming.phases.resident_seconds = 0.3;
    second.data.streaming.phases.router_to_resident_seconds = 0.1;

    history.observe(&first);
    history.observe(&second);

    assert!((history.samples.back().unwrap().values[2].unwrap() - 75.0).abs() < 1e-10);

    second.observation.observed_at_uptime_seconds += 1.0;
    second.observation.elapsed_seconds += 1.0;

    history.observe(&second);

    assert_eq!(
        history.samples.back().unwrap().values,
        [Some(0.0), Some(0.0), None]
    );
}

#[test]
fn reconnect_leaves_a_gap_and_repeated_snapshots_add_nothing() {
    let mut history = recorded(&[(10.0, 0), (11.0, 1_000_000)]);

    history.disconnect();
    history.observe(&observation(11.0, 1_000_000));

    assert_eq!(history.samples.len(), 2);

    history.observe(&observation(20.0, 10_000_000));
    assert_eq!(history.samples.back().unwrap().values, [None; 3]);
    history.observe(&observation(21.0, 12_000_000));
    assert_eq!(history.samples.back().unwrap().values[0], Some(2.0));
}

#[test]
fn counter_regression_and_model_reload_reset_history() {
    let mut history = recorded(&[(10.0, 10), (11.0, 20)]);

    history.observe(&observation(12.0, 5));
    assert_eq!(history.samples.len(), 1);

    let mut reloaded = observation(13.0, 30);
    reloaded.observation.elapsed_seconds = 1.0;

    history.observe(&reloaded);
    assert_eq!(history.samples.len(), 1);
    assert_eq!(history.samples.back().unwrap().values, [None; 3]);
}

#[test]
fn history_is_bounded_by_count_and_time() {
    let mut history = History::default();

    for time in 2..500 {
        history.observe(&observation(f64::from(time), time as u64));
    }

    assert_eq!(history.samples.len(), 120);
    history.observe(&observation(1000.0, 1000));
    assert_eq!(history.samples.len(), 1);
}

#[test]
fn invalid_observation_cannot_poison_chart_bounds() {
    let mut history = History::default();
    let mut bad = observation(11.0, 10);
    bad.data.streaming.phases.demand_wait_seconds = f64::NAN;

    history.observe(&observation(10.0, 0));
    history.observe(&bad);
    history.observe(&observation(12.0, 20));

    assert_eq!(history.samples.back().unwrap().values, [None; 3]);
}
