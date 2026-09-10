use crate::{
    config::Config,
    server::sessions::{Store, Turn},
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

pub(in crate::server) fn new_session(
    config: &Config,
    settings: Value,
) -> (Arc<Mutex<Store>>, String) {
    let mut store = Store::new(config);
    let id = store
        .create(&settings, config)
        .expect("create test session");

    (Arc::new(Mutex::new(store)), id)
}

pub(in crate::server) fn message(session: &str, text: &str) -> Value {
    json!({"session_id":session,"messages":[{"role":"user","content":text}]})
}

pub(in crate::server) fn begin_turn(
    store: &Arc<Mutex<Store>>,
    body: &Value,
    request: &str,
) -> Turn {
    Turn::begin(store, body, request, 4096)
        .expect("begin test turn")
        .expect("retained session turn")
}
