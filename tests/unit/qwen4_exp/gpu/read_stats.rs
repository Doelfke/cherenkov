use super::*;

#[test]
fn reads_keep_precision_source_failure_and_duration_separate() {
    let tracker = ReadTracker::new(2);
    let q4 = tracker.ticket(0, 0, ReadSource::Prefetch, 100);
    let q2 = tracker.ticket(0, 2, ReadSource::Demand, 40);

    assert!(!q4.done());
    q4.finish(100, 0.02);
    q2.finish(20, 0.03);

    let snapshot = tracker.snapshot();

    assert_eq!(snapshot[0][0].prefetch.completed_reads, 1);
    assert_eq!(snapshot[0][0].prefetch.completed_bytes, 100);
    assert_eq!(snapshot[0][0].prefetch.read_seconds, 0.02);
    assert_eq!(snapshot[0][2].demand.failed_reads, 1);
    assert_eq!(snapshot[0][2].demand.completed_reads, 0);
    assert_eq!(snapshot[0][2].demand.completed_bytes, 20);
    assert_eq!(snapshot[1][0].total().requested_reads, 0);
    assert!(q4.done());
}

#[test]
fn concurrent_reads_accumulate_and_snapshot_does_not_change_later() {
    let tracker = ReadTracker::new(1);
    let before = tracker.snapshot();

    std::thread::scope(|scope| {
        for _ in 0..16 {
            let ticket = tracker.ticket(0, 1, ReadSource::Prefill, 64);

            scope.spawn(move || ticket.finish(64, 0.125));
        }
    });

    let after = tracker.snapshot()[0][1].prefill;

    assert_eq!(before[0][1].prefill.completed_reads, 0);
    assert_eq!(after.completed_reads, 16);
    assert_eq!(after.completed_bytes, 1024);
    assert_eq!(after.read_seconds, 2.0);
    assert_eq!(after.max_read_seconds, 0.125);
}
