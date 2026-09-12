use super::*;

#[test]
fn measured_slot_and_mapped_reads_report_short_reads() {
    let file = tempfile::tempfile().unwrap();

    use std::os::unix::fs::FileExt;

    file.write_all_at(&[7; 32], 0).unwrap();

    let tracker = ReadTracker::new(1);
    let mut buffer = [0u8; 64];
    let ticket = tracker.ticket(0, 2, ReadSource::Demand, 64);

    fetch_into_slots(
        &file,
        &[RecordRead {
            destination: buffer.as_mut_ptr() as usize,
            file_offset: 0,
            bytes: 64,
            ticket: Some(ticket),
        }],
    );
    assert_eq!(&buffer[..32], &[7; 32]);
    assert_eq!(&buffer[32..], &[0; 32]);

    let stats = tracker.snapshot()[0][2].demand;

    assert_eq!(stats.completed_bytes, 32);
    assert_eq!(stats.failed_reads, 1);
    assert_eq!(read_record(&file, 0, 0, 32), 32);
    assert_eq!(read_record(&file, 0, 0, 64), 32);
}

#[test]
fn reordered_reads_keep_their_own_precision_and_completion_ticket() {
    use std::os::unix::fs::FileExt;

    let file = tempfile::tempfile().unwrap();

    file.write_all_at(&[7; 16], 0).unwrap();

    let tracker = ReadTracker::new(2);
    let mut full = [0u8; 8];
    let mut short = [0u8; 8];
    let full_ticket = tracker.ticket(0, 0, ReadSource::Prefill, 8);
    let short_ticket = tracker.ticket(1, 2, ReadSource::Prefill, 8);
    let mut reads = vec![
        RecordRead {
            destination: full.as_mut_ptr() as usize,
            file_offset: 0,
            bytes: 8,
            ticket: Some(full_ticket.clone()),
        },
        RecordRead {
            destination: short.as_mut_ptr() as usize,
            file_offset: 12,
            bytes: 8,
            ticket: Some(short_ticket.clone()),
        },
    ];

    reads.reverse();
    fetch_into_slots(&file, &reads);

    let snapshot = tracker.snapshot();

    assert_eq!(full, [7; 8]);
    assert_eq!(short, [7, 7, 7, 7, 0, 0, 0, 0]);
    assert!(full_ticket.done());
    assert!(short_ticket.done());
    assert_eq!(snapshot[0][0].prefill.completed_reads, 1);
    assert_eq!(snapshot[1][2].prefill.failed_reads, 1);
    assert_eq!(snapshot[1][2].prefill.completed_bytes, 4);
}

#[test]
fn tracked_plan_finishes_and_keeps_low_bit_file_selection() {
    let base = tempfile::tempfile().unwrap();
    let low = tempfile::tempfile().unwrap();

    use std::os::unix::fs::FileExt;

    base.write_all_at(&[4; 16], 0).unwrap();
    low.write_all_at(&[2; 16], 0).unwrap();

    let tracker = ReadTracker::new(1);
    let mut output = [0u8; 16];
    let mut plan = ReadPlan {
        items: vec![PlannedRead {
            transfer: RecordRead {
                ticket: None,
                destination: output.as_mut_ptr() as usize,
                file_offset: 0,
                bytes: 16,
            },
            kind: 2,
            need_index: 0,
        }],
        low_file: Some(low),
    };

    plan.observe(&tracker, 0, ReadSource::Demand);

    let flags = plan.run_tracked(&base, &base, 1);

    flags[0].wait().unwrap();
    assert_eq!(output, [2; 16]);
    assert_eq!(tracker.snapshot()[0][2].demand.completed_reads, 1);
    assert_eq!(tracker.snapshot()[0][0].demand.completed_reads, 0);
}
