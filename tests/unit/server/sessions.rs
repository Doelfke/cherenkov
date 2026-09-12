use super::*;
use crate::server::tests::fixtures::{begin_turn, message, new_session};
use rand::SeedableRng;

#[test]
fn cancelled_turn_does_not_change_history_settings_or_rng() {
    let config = Config::default();
    let (store, id) = new_session(&config, json!({"temperature":0.7}));
    let original = store.lock().unwrap().show(&id).unwrap();
    let original_rng = StdRng::seed_from_u64(13);
    store.lock().unwrap().entries.get_mut(&id).unwrap().rng = Some(original_rng.clone());
    let mut body = message(&id, "Hello");
    body["temperature"] = json!(1.2);
    let turn = begin_turn(&store, &body, "req-1");
    let prepared = turn
        .prepare(
            "unfinished",
            Some(StdRng::seed_from_u64(99)),
            Default::default(),
            None,
        )
        .unwrap();

    drop(prepared); // Cancellation after a step, before publishing the final response.
    assert_eq!(store.lock().unwrap().show(&id).unwrap(), original);
    assert_eq!(store.lock().unwrap().entries[&id].rng, Some(original_rng));
}

#[test]
fn completed_turn_retains_sampling_and_incremental_history() {
    let config = Config::default();
    let (store, id) = new_session(&config, json!({"temperature":0.7}));
    let body = message(&id, "Hello");
    let turn = begin_turn(&store, &body, "req-1");

    assert_eq!(turn.input.sampling.temperature, 0.7);
    assert!(Turn::begin(&store, &body, "req-2", 4096).is_err());
    turn.prepare("Hi", None, Default::default(), None)
        .unwrap()
        .publish();

    let next = message(&id, "Continue");
    let turn = begin_turn(&store, &next, "req-3");

    assert_eq!(turn.input.messages.len(), 3);
    assert_eq!(turn.input.sampling.temperature, 0.7);
}

#[test]
fn idle_sessions_are_evicted_but_active_sessions_are_pinned() {
    let mut config = Config::default();
    config.limits.max_sessions = 1;
    let (store, id) = new_session(&config, json!({}));
    let body = message(&id, "Hello");
    let turn = begin_turn(&store, &body, "req-1");

    assert!(store.lock().unwrap().create(&json!({}), &config).is_err());
    drop(turn);
    store.lock().unwrap().create(&json!({}), &config).unwrap();
    assert!(store.lock().unwrap().show(&id).is_err());
    assert_eq!(store.lock().unwrap().count(), 1);
}

#[test]
fn random_stream_continues_across_turns_and_greedy_does_not_reset_it() {
    let config = Config::default();
    let (store, id) = new_session(&config, json!({"temperature":0.7,"seed":42}));
    let mut rng = StdRng::seed_from_u64(42);

    for _ in 0..7 {
        let _ = rand::Rng::random::<f64>(&mut rng);
    }

    let body = message(&id, "One");
    let turn = begin_turn(&store, &body, "one");

    turn.prepare("First", Some(rng.clone()), Default::default(), None)
        .unwrap()
        .publish();

    let mut body = message(&id, "Two");
    body["temperature"] = json!(0);
    let turn = begin_turn(&store, &body, "two");

    assert_eq!(turn.rng, Some(rng.clone()));
    turn.prepare("Second", None, Default::default(), None)
        .unwrap()
        .publish();

    let mut body = message(&id, "Three");
    body["temperature"] = json!(0.7);
    let turn = begin_turn(&store, &body, "three");

    assert_eq!(turn.rng, Some(rng.clone()));
    drop(turn);

    // Only an explicit seed restarts a continued session's stream.
    body["seed"] = json!(99);
    let turn = begin_turn(&store, &body, "restart");

    assert!(turn.rng.is_none());
    assert_eq!(turn.input.sampling.seed, Some(99));
    drop(turn);
    assert_eq!(store.lock().unwrap().entries[&id].rng, Some(rng));
}

#[test]
fn history_budget_failure_releases_the_pin_and_reservation() {
    let config = Config::default();
    let (store, id) = new_session(&config, json!({}));
    let initial = store.lock().unwrap().bytes();
    store.lock().unwrap().capacity = initial + 64;
    let body = message(&id, "Hello");
    let turn = begin_turn(&store, &body, "large");

    assert!(
        turn.prepare(&"x".repeat(256), None, Default::default(), None)
            .is_err()
    );

    let mut guard = store.lock().unwrap();

    assert_eq!(guard.bytes(), initial);

    let shown = guard.show(&id).unwrap();

    assert_eq!(shown["messages"], json!([]));
    assert!(shown["active_request_id"].is_null());
}

#[test]
fn expiry_and_context_validation_preserve_active_sessions() {
    let mut config = Config::default();
    config.limits.session_idle_seconds = 10;
    let (store, id) = new_session(&config, json!({"context_tokens":64}));
    let mut body = message(&id, "Hello");
    body["context_tokens"] = json!(65);

    assert!(Turn::begin(&store, &body, "invalid", 4096).is_err());

    body["context_tokens"] = json!(64);
    let turn = begin_turn(&store, &body, "active");
    store.lock().unwrap().entries.get_mut(&id).unwrap().touched =
        Instant::now() - Duration::from_secs(11);

    assert!(store.lock().unwrap().show(&id).is_ok());
    drop(turn);
    assert!(store.lock().unwrap().show(&id).is_err());
    assert_eq!(store.lock().unwrap().count(), 0);
}

#[test]
fn deleting_an_expired_session_returns_not_found() {
    let mut config = Config::default();
    config.limits.session_idle_seconds = 1;
    let (store, id) = new_session(&config, json!({}));
    let mut store = store.lock().unwrap();
    store.entries.get_mut(&id).unwrap().touched = Instant::now() - Duration::from_secs(2);
    let error = store.delete(&id).unwrap_err();

    assert_eq!(crate::server::failure::status(&error), 404);
    assert_eq!(store.count(), 0);
}
