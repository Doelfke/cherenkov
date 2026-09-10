//! HTTP endpoints and admission into the generation queue.

use super::http::{error, read_http, respond};
use super::{ApiKind, Job, MODEL, failure, registry, sessions};
use crate::control::State;
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    net::TcpStream,
    sync::{Arc, mpsc},
    time::Duration,
};

pub(super) fn connection(
    mut stream: TcpStream,
    queue: &mpsc::SyncSender<Job>,
    state: &State,
    requests: &Arc<registry::Registry>,
    sessions: &Arc<std::sync::Mutex<sessions::Store>>,
) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    stream.set_nodelay(true)?;

    let settings = state.config();
    let (method, path, body) = match read_http(&mut stream, settings.config.limits.request_bytes) {
        Ok(r) => r,
        Err(e) => return error(&mut stream, 400, &e.to_string()),
    };

    if let Some(result) =
        session_endpoint(&method, &path, &body, &settings.config, requests, sessions)
    {
        return match result {
            Ok((status, body)) => respond(&mut stream, status, &body),
            Err(e) => request_error(&mut stream, state, &e),
        };
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, &json!({"status":"ok"})),
        ("GET", "/v1/models") => respond(
            &mut stream,
            200,
            &json!({
                "object":"list", "data":[{"id":MODEL,"object":"model","created":0,"owned_by":"local"}]
            }),
        ),
        ("POST", "/v1/chat/completions" | "/v1/completions") => {
            let body: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => return error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };

            if !body.is_object() {
                return error(&mut stream, 400, "request must be an object");
            }

            if body.get("request_id").is_some_and(|v| !v.is_string()) {
                return error(&mut stream, 400, "request_id must be a string");
            }

            let ticket = match requests.register(
                body["request_id"].as_str(),
                settings.config.limits.queued_requests + settings.config.limits.active_requests,
            ) {
                Ok(ticket) => ticket,
                Err(e) => return request_error(&mut stream, state, &e),
            };

            if path == "/v1/completions" && !body["session_id"].is_null() {
                return error(
                    &mut stream,
                    400,
                    "retained sessions require chat completions",
                );
            }

            let session = match sessions::Turn::begin(
                sessions,
                &body,
                &ticket.id,
                settings.config.limits.request_bytes,
            ) {
                Ok(turn) => turn,
                Err(e) => return request_error(&mut stream, state, &e),
            };
            let job = Job {
                stream,
                body,
                settings,
                ticket,
                session,
                kind: if path == "/v1/chat/completions" {
                    ApiKind::Chat
                } else {
                    ApiKind::Completion
                },
            };

            state.update(|s| s.queued_requests += 1);

            let (mut job, message) = match queue.try_send(job) {
                Ok(()) => return Ok(()),
                Err(mpsc::TrySendError::Full(job)) => (job, "generation queue is full"),
                Err(mpsc::TrySendError::Disconnected(job)) => (job, "model unavailable"),
            };

            state.update(|s| {
                s.queued_requests -= 1;
                s.rejected_requests += 1;
            });

            error(&mut job.stream, 503, message)
        }
        _ => error(&mut stream, 404, "unknown endpoint"),
    }
}

pub(super) fn session_endpoint(
    method: &str,
    path: &str,
    body: &[u8],
    config: &crate::config::Config,
    requests: &registry::Registry,
    sessions: &std::sync::Mutex<sessions::Store>,
) -> Option<Result<(u16, Value)>> {
    if method == "GET" && path == "/v1/requests" {
        return Some(Ok((200, requests.list())));
    }

    if let Some(id) = path
        .strip_prefix("/v1/requests/")
        .and_then(|p| p.strip_suffix("/cancel"))
    {
        if method != "POST" {
            return None;
        }

        return Some(Ok((
            if requests.cancel(id) { 200 } else { 404 },
            json!({"id":id}),
        )));
    }

    if method == "POST" && path == "/v1/sessions" {
        return Some(
            serde_json::from_slice(body)
                .map_err(Into::into)
                .and_then(|v| {
                    let mut store = sessions.lock().unwrap();
                    let id = store.create(&v, config)?;

                    store.show(&id)
                })
                .map(|v| (200, v)),
        );
    }

    let id = path.strip_prefix("/v1/sessions/")?;

    match method {
        "GET" => Some(sessions.lock().unwrap().show(id).map(|v| (200, v))),
        "DELETE" => Some(
            sessions
                .lock()
                .unwrap()
                .delete(id)
                .map(|()| (200, json!({"id":id,"deleted":true}))),
        ),
        _ => None,
    }
}

fn request_error(out: &mut TcpStream, state: &State, cause: &anyhow::Error) -> Result<()> {
    let status = failure::status(cause);

    state.update(|s| s.rejected_requests += u64::from(status == 503));

    error(out, status, &cause.to_string())
}
