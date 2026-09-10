use super::*;

#[test]
fn cancellation_wins_before_publication_and_never_after_it() {
    let registry = Arc::new(Registry::default());
    let request = registry.register(Some("before-step"), 2).unwrap();
    assert!(registry.cancel(&request.id));
    assert!(request.cancelled());
    assert!(!request.complete());
    let request = registry.register(Some("after-step"), 2).unwrap();
    // A completed GPU step has not published the response yet.
    assert!(registry.cancel(&request.id));
    assert!(!request.complete());
    drop(request);
    let request = registry.register(Some("published"), 2).unwrap();
    assert!(request.complete());
    assert!(!registry.cancel(&request.id));
}

#[test]
fn registry_is_bounded_and_ids_remain_pinned_through_output() {
    let registry = Arc::new(Registry::default());
    let ticket = registry.register(Some("live"), 1).unwrap();
    assert!(registry.register(Some("other"), 1).is_err());
    let writer = ticket.clone();
    drop(ticket);
    assert!(registry.cancel("live"));
    drop(writer);
    assert!(registry.register(Some("live"), 1).is_ok());
}

#[test]
fn simultaneous_cancel_and_complete_have_one_winner() {
    use std::sync::Barrier;
    for _ in 0..64 {
        let registry = Arc::new(Registry::default());
        let ticket = registry.register(Some("race"), 1).unwrap();
        let barrier = Barrier::new(2);
        let (cancelled, completed) = std::thread::scope(|scope| {
            let cancel = scope.spawn(|| {
                barrier.wait();
                registry.cancel(&ticket.id)
            });
            barrier.wait();
            let completed = ticket.complete();
            (cancel.join().unwrap(), completed)
        });
        assert_ne!(cancelled, completed);
        assert_eq!(ticket.cancelled(), cancelled);
    }
}

#[test]
fn invalid_duplicate_and_exhausted_ids_have_distinct_statuses() {
    let registry = Arc::new(Registry::default());
    for id in [
        "".to_owned(),
        "white space".into(),
        "line\nbreak".into(),
        "x".repeat(81),
    ] {
        let error = registry.register(Some(&id), 1).err().expect("invalid ID");
        assert_eq!(super::super::failure::status(&error), 400);
    }
    let live = registry.register(Some("live"), 1).unwrap();
    for (id, expected) in [("live", 409), ("other", 503)] {
        let error = registry
            .register(Some(id), 1)
            .err()
            .expect("registration rejected");
        assert_eq!(super::super::failure::status(&error), expected);
    }
    assert!(registry.cancel(&live.id));
    assert!(registry.cancel(&live.id)); // Repeated cancellation is idempotent while registered.
    drop(live);
    assert!(!registry.cancel("live"));
    assert_eq!(registry.list()["data"], json!([]));
}
