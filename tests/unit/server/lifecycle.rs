use super::fixtures;
use crate::config::Config;
use crate::server::{failure, registry, routes::session_endpoint, sessions};
use fixtures::{begin_turn, message, new_session};
use serde_json::json;

#[test]
fn active_sessions_reject_deletion_reuse_and_eviction_until_the_turn_drops() {
    let mut config = Config::default();
    config.limits.max_sessions = 1;
    let (store, id) = new_session(&config, json!({}));
    let registry = registry::Registry::default();
    let body = message(&id, "Hello");
    let turn = begin_turn(&store, &body, "pinned");
    let path = format!("/v1/sessions/{id}");
    for (method, path, bytes, expected) in [
        ("DELETE", path.as_str(), b"".as_slice(), 409),
        ("POST", "/v1/sessions", b"{}".as_slice(), 503),
    ] {
        let error = session_endpoint(method, path, bytes, &config, &registry, &store)
            .expect("session route")
            .unwrap_err();
        assert_eq!(failure::status(&error), expected);
    }
    let error = sessions::Turn::begin(&store, &body, "duplicate", 4096)
        .err()
        .expect("session already pinned");
    assert_eq!(failure::status(&error), 409);
    drop(turn);
    let (status, deleted) = session_endpoint("DELETE", &path, b"", &config, &registry, &store)
        .expect("session route")
        .unwrap();
    assert_eq!(status, 200);
    assert_eq!(deleted, json!({"id":id,"deleted":true}));
    let error = session_endpoint("GET", &path, b"", &config, &registry, &store)
        .expect("session route")
        .unwrap_err();
    assert_eq!(failure::status(&error), 404);
}

#[test]
fn new_defaults_do_not_change_existing_sessions_sampling() {
    let mut config = Config::default();
    config.defaults.sampling.temperature = 0.7;
    let (store, existing) = new_session(&config, json!({"seed":42}));
    config.defaults.sampling.temperature = 1.2;
    let new = store.lock().unwrap().create(&json!({}), &config).unwrap();
    let previous = message(&existing, "Continue");
    let previous = begin_turn(&store, &previous, "previous-defaults");
    let current = message(&new, "Hello");
    let current = begin_turn(&store, &current, "current-defaults");
    assert_eq!(previous.input.sampling.temperature, 0.7);
    assert_eq!(previous.input.sampling.seed, Some(42));
    assert_eq!(current.input.sampling.temperature, 1.2);
    assert!(current.input.sampling.seed.is_none());
}
