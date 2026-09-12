//! Bounded request registration and cancellation, independent of the GPU worker.

use super::failure::Failure;
use anyhow::{Result, ensure};
use rand::Rng;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
};

const LIVE: u8 = 0;
const CANCELLED: u8 = 1;
const COMPLETED: u8 = 2;

pub(super) fn id(prefix: &str) -> String {
    format!("{prefix}-{:032x}", rand::rng().random::<u128>())
}

/// OpenAI-style identifier minted for one decoded model tool call.
pub(super) fn tool_call_id() -> String {
    format!("call_{:016x}", rand::rng().random::<u64>())
}

#[derive(Default)]
pub(super) struct Registry {
    entries: Mutex<HashMap<String, Arc<AtomicU8>>>,
}

impl Registry {
    pub(super) fn register(
        self: &Arc<Self>,
        requested: Option<&str>,
        limit: usize,
    ) -> Result<Arc<Ticket>> {
        let id = requested.map(str::to_owned).unwrap_or_else(|| id("req"));

        ensure!(
            !id.is_empty()
                && id.len() <= 80
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "request_id must be 1..80 ASCII letters, digits, '-' or '_'"
        );

        let mut entries = self.entries.lock().unwrap();

        ensure!(
            !entries.contains_key(&id),
            Failure(409, "request_id is already active")
        );
        ensure!(
            entries.len() < limit,
            Failure(503, "request capacity exhausted")
        );

        let flag = Arc::new(AtomicU8::new(LIVE));

        entries.insert(id.clone(), flag.clone());

        Ok(Arc::new(Ticket {
            id,
            flag,
            registry: self.clone(),
        }))
    }

    pub(super) fn cancel(&self, id: &str) -> bool {
        let entries = self.entries.lock().unwrap();
        let Some(flag) = entries.get(id) else {
            return false;
        };

        matches!(
            flag.compare_exchange(LIVE, CANCELLED, Ordering::AcqRel, Ordering::Acquire),
            Ok(_) | Err(CANCELLED)
        )
    }

    pub(super) fn list(&self) -> Value {
        let entries = self.entries.lock().unwrap();
        let mut ids: Vec<_> = entries.keys().collect();

        ids.sort();

        let requests: Vec<_> = ids
            .into_iter()
            .map(|id| {
                json!({
                    "id": id,
                    "cancel_requested": entries[id].load(Ordering::Acquire) == CANCELLED,
                })
            })
            .collect();

        json!({"object": "list", "data": requests})
    }
}

pub(super) struct Ticket {
    pub id: String,
    flag: Arc<AtomicU8>,
    registry: Arc<Registry>,
}

impl Ticket {
    pub(super) fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire) == CANCELLED
    }

    pub(super) fn cancel(&self) {
        let _ = self.registry.cancel(&self.id);
    }

    /// Cancellation and final publication have one atomic ordering boundary.
    pub(super) fn complete(&self) -> bool {
        self.flag
            .compare_exchange(LIVE, COMPLETED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.registry.entries.lock().unwrap().remove(&self.id);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/server/registry.rs"]
mod tests;
