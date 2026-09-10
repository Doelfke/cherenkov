//! Bounded conversation history, committed at the final-response publication boundary.

use super::{failure::Failure, registry, request::SessionInput};
use crate::units::BYTES_PER_MIB;
use crate::{config::Config, sampling::Sampling};
use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

struct Session {
    history: Box<str>,
    sampling: Sampling,
    context: usize,
    rng: Option<StdRng>,
    busy: Option<String>,
    touched: Instant,
    reserved: usize,
}

pub(super) struct Store {
    entries: HashMap<String, Session>,
    limit: usize,
    capacity: usize,
    idle: Duration,
    pub evictions: u64,
}

impl Store {
    pub(super) fn new(config: &Config) -> Self {
        Self {
            entries: HashMap::new(),
            limit: config.limits.max_sessions,
            capacity: config.limits.session_history_mib * BYTES_PER_MIB,
            idle: Duration::from_secs(config.limits.session_idle_seconds),
            evictions: 0,
        }
    }

    pub(super) fn bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(id, session)| {
                id.capacity() + session.history.len() + session.reserved + size_of::<Session>()
            })
            .sum()
    }

    pub(super) fn count(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn expire(&mut self) {
        if self.idle.is_zero() {
            return;
        }

        self.entries
            .retain(|_, session| session.busy.is_some() || session.touched.elapsed() < self.idle);
    }

    fn make_room(&mut self, extra: usize, create: bool, protected: Option<&str>) -> Result<()> {
        ensure!(
            extra <= self.capacity,
            "session history exceeds its memory budget"
        );

        while self.bytes() + extra > self.capacity || (create && self.entries.len() >= self.limit) {
            let oldest = self
                .entries
                .iter()
                .filter(|(id, session)| session.busy.is_none() && Some(id.as_str()) != protected)
                .min_by_key(|(_, session)| session.touched)
                .map(|(id, _)| id.clone())
                .ok_or(Failure(503, "all session storage is in use"))?;

            self.entries.remove(&oldest);

            self.evictions += 1;
        }

        Ok(())
    }

    pub(super) fn create(&mut self, body: &Value, config: &Config) -> Result<String> {
        ensure!(self.limit > 0, "retained sessions are disabled");
        ensure!(body.is_object(), "session must be an object");

        let sampling = Sampling::from_request(body, &config.defaults.sampling)?;
        let context = body
            .get("context_tokens")
            .map(|v| v.as_u64().context("context_tokens must be an integer"))
            .transpose()?
            .unwrap_or(config.limits.context_tokens as u64) as usize;

        ensure!(
            context > 0 && context <= config.limits.context_tokens,
            "context_tokens exceeds server capacity"
        );
        self.expire();

        let id = registry::id("session");

        self.make_room(size_of::<Session>() + id.capacity() + 2, true, None)?;
        self.entries.insert(
            id.clone(),
            Session {
                history: "[]".into(),
                sampling,
                context,
                rng: None,
                busy: None,
                touched: Instant::now(),
                reserved: 0,
            },
        );

        Ok(id)
    }

    pub(super) fn show(&mut self, id: &str) -> Result<Value> {
        self.expire();

        let session = self
            .entries
            .get(id)
            .ok_or(Failure(404, "unknown or expired session"))?;

        Ok(json!({
            "id": id,
            "object": "session",
            "sampling": session.sampling,
            "context_tokens": session.context,
            "active_request_id": session.busy,
            "messages": serde_json::from_str::<Value>(&session.history)?,
        }))
    }

    pub(super) fn delete(&mut self, id: &str) -> Result<()> {
        self.expire();

        let session = self
            .entries
            .get(id)
            .ok_or(Failure(404, "unknown or expired session"))?;

        ensure!(
            session.busy.is_none(),
            Failure(409, "cancel the active request before deleting its session")
        );
        self.entries.remove(id);

        Ok(())
    }
}

/// Pin a session through admission, prefill and decode. Drop rolls back cancelled turns.
pub(super) struct Turn {
    pub id: String,
    store: Arc<Mutex<Store>>,
    pub input: SessionInput,
    pub rng: Option<StdRng>,
}

impl Turn {
    pub(super) fn begin(
        store: &Arc<Mutex<Store>>,
        body: &Value,
        request_id: &str,
        request_bytes: usize,
    ) -> Result<Option<Self>> {
        let Some(id) = body.get("session_id").filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        let id = id
            .as_str()
            .context("session_id must be a string")?
            .to_owned();
        let incoming = body["messages"]
            .as_array()
            .context("sessions require chat messages")?
            .clone();

        ensure!(!incoming.is_empty(), "messages must not be empty");

        let mut guard = store.lock().unwrap();

        guard.expire();

        let session = guard
            .entries
            .get_mut(&id)
            .ok_or(Failure(404, "unknown or expired session"))?;

        ensure!(
            session.busy.is_none(),
            Failure(409, "session already has an active request")
        );

        let sampling = Sampling::from_request(body, &session.sampling)?;
        let restart_rng = body.get("seed").is_some_and(|v| !v.is_null());
        let mut messages: Vec<Value> = serde_json::from_str(&session.history)?;

        messages.extend(incoming);
        ensure!(
            serde_json::to_vec(&messages)?.len() <= request_bytes,
            "session history and new messages exceed request_bytes"
        );

        let context = body
            .get("context_tokens")
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .context("context_tokens must be a positive integer")
            })
            .transpose()?
            .unwrap_or(session.context);

        ensure!(
            context > 0 && context <= session.context,
            "context_tokens exceeds session capacity"
        );

        // Explicit seeds restart the stream. Otherwise an existing session continues it.
        let rng = if restart_rng {
            None
        } else {
            session.rng.clone()
        };
        session.busy = Some(request_id.to_owned());

        Ok(Some(Self {
            id,
            store: store.clone(),
            input: SessionInput {
                messages,
                sampling,
                context,
            },
            rng,
        }))
    }

    pub(super) fn prepare(mut self, text: &str, rng: Option<StdRng>) -> Result<Commit> {
        self.input
            .messages
            .push(json!({"role":"assistant", "content":text}));

        let history = serde_json::to_string(&self.input.messages)?.into_boxed_str();
        let mut store = self.store.lock().unwrap();
        let previous_bytes = store
            .entries
            .get(&self.id)
            .context("session disappeared")?
            .history
            .len();
        let growth = history.len().saturating_sub(previous_bytes);

        store.make_room(growth, false, Some(&self.id))?;

        store
            .entries
            .get_mut(&self.id)
            .expect("pinned session")
            .reserved = growth;

        drop(store);

        // Greedy turns consume no draws; retain the previous random stream.
        let rng = rng.or_else(|| self.rng.clone());

        Ok(Commit {
            turn: self,
            history,
            rng,
        })
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        if let Some(session) = self.store.lock().unwrap().entries.get_mut(&self.id) {
            session.busy = None;
            session.reserved = 0;
        }
    }
}

pub(super) struct Commit {
    turn: Turn,
    history: Box<str>,
    rng: Option<StdRng>,
}

impl Commit {
    pub(super) fn publish(self) {
        let mut store = self.turn.store.lock().unwrap();
        let session = store
            .entries
            .get_mut(&self.turn.id)
            .expect("pinned session");
        session.history = self.history;
        session.sampling = self.turn.input.sampling.clone();
        session.rng = self.rng;
        session.reserved = 0;
        session.touched = Instant::now();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/server/sessions.rs"]
mod tests;
