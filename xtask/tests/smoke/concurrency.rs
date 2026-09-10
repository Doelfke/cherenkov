use super::*;
use anyhow::Context;
use std::sync::mpsc;

#[path = "concurrency/isolation.rs"]
mod isolation;
#[path = "concurrency/limits.rs"]
mod limits;

fn completion_body(session: &str, prompt: &str, max_tokens: usize) -> Value {
    json!({
        "session_id": session,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })
}

fn wait_idle(server: &Server) -> Result<()> {
    wait_for(
        || {
            let stats = server.stats()?;
            let requests = request(&server.address, "/v1/requests", None)?.success()?;
            Ok(stats["active_requests"] == 0
                && stats["queued_requests"] == 0
                && stats["active_state_reserved_bytes"] == 0
                && requests["data"] == json!([]))
        },
        Duration::from_secs(30),
    )
}

fn session(address: &str, seed: u64) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct CreatedSession {
        id: String,
    }

    let response = request(
        address,
        "/v1/sessions",
        Some(&json!({"temperature":0.7,"top_k":20,"seed":seed})),
    )?
    .success()?;
    let created: CreatedSession = serde_json::from_value(response)?;
    Ok(created.id)
}

fn session_pair(active: usize) -> Result<(Server, [String; 2])> {
    let server = Server::start_with_active(900, active)?;
    let ids = [session(&server.address, 42)?, session(&server.address, 71)?];
    Ok((server, ids))
}

fn cancel(address: &str, id: &str) -> Result<()> {
    request(
        address,
        &format!("/v1/requests/{id}/cancel"),
        Some(&json!({})),
    )?
    .success()?;
    Ok(())
}

fn stored_session(address: &str, id: &str) -> Result<Value> {
    request(address, &format!("/v1/sessions/{id}"), None)?.success()
}

fn stream_events(
    address: &str,
    body: Value,
    tag: &'static str,
    sender: mpsc::Sender<(&'static str, Value)>,
) -> Result<Vec<Value>> {
    let mut stream = EventStream::open(address, &body)?;
    let mut events = Vec::new();
    while let Some(value) = stream.next()? {
        sender.send((tag, value.clone()))?;
        events.push(value);
    }
    Ok(events)
}

fn stream_body(session: &str, request: &str, max_tokens: usize) -> Value {
    json!({
        "session_id": session,
        "request_id": request,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Explain how hash tables work in detail."}],
    })
}

fn wait_for_both_streams(receiver: &mpsc::Receiver<(&str, Value)>) -> Result<()> {
    let mut progress = [0; 2];
    while progress.iter().any(|&n| n < 2) {
        let (id, event) = receiver.recv_timeout(Duration::from_secs(180))?;
        anyhow::ensure!(event.get("error").is_none(), "{event}");
        anyhow::ensure!(
            event["choices"][0]["finish_reason"].is_null(),
            "a stream finished before both made progress"
        );
        if event["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
        {
            progress[usize::from(id == "b")] += 1;
        }
    }
    Ok(())
}

fn assert_continued_session(server: &Server, id: &str) -> Result<()> {
    let continued = request(
        &server.address,
        "/v1/chat/completions",
        Some(&json!({"session_id":id,"max_tokens":4,
            "messages":[{"role":"user","content":"Give an example."}]})),
    )?
    .success()?;
    assert_eq!(continued["usage"]["completion_tokens"], 4);
    let history = stored_session(&server.address, id)?;
    assert_eq!(history["messages"].as_array().unwrap().len(), 4);
    assert_eq!(history["sampling"]["seed"], 71);
    assert_eq!(history["sampling"]["temperature"], 0.7);
    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn interleaved_sessions_cancel_without_affecting_the_other_stream() -> Result<()> {
    let (server, [a, b]) = session_pair(2)?;
    let (sender, receiver) = mpsc::channel();
    let (left, right) = std::thread::scope(|scope| -> Result<_> {
        let left = scope.spawn(|| {
            stream_events(
                &server.address,
                stream_body(&a, "interleave-a", 48),
                "a",
                sender.clone(),
            )
        });
        let right = scope.spawn(|| {
            stream_events(
                &server.address,
                stream_body(&b, "interleave-b", 16),
                "b",
                sender.clone(),
            )
        });
        wait_for_both_streams(&receiver)?;
        cancel(&server.address, "interleave-a")?;
        Ok((left.join().unwrap()?, right.join().unwrap()?))
    })?;
    assert!(
        left.iter()
            .any(|e| e["choices"][0]["finish_reason"] == "cancelled")
    );
    assert!(
        right
            .iter()
            .any(|e| e["choices"][0]["finish_reason"] == "length")
    );
    assert_eq!(right.last().unwrap()["usage"]["completion_tokens"], 16);
    assert!(
        left.last().unwrap()["usage"]["completion_tokens"]
            .as_u64()
            .unwrap()
            < 48
    );
    let cancelled = stored_session(&server.address, &a)?;
    let completed = stored_session(&server.address, &b)?;
    assert_eq!(cancelled["messages"], json!([]));
    assert_eq!(completed["messages"].as_array().unwrap().len(), 2);
    assert_eq!(completed["sampling"]["temperature"], 0.7);
    let retry = request(
        &server.address,
        "/v1/chat/completions",
        Some(&json!({
        "session_id":a, "max_tokens":4,
        "messages":[{"role":"user","content":"Explain how hash tables work in detail."}]})),
    )?
    .success()?;
    assert_eq!(retry["usage"]["completion_tokens"], 4);
    assert!(
        retry["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    wait_for(
        || Ok(server.stats()?["active_state_reserved_bytes"] == 0),
        Duration::from_secs(10),
    )?;
    assert_continued_session(&server, &b)?;
    save_result(
        "sessions",
        json!({"interleaving":"both streams progressed","cancelled":left.last(),
        "completed":right.last(),"retry":retry["usage"],"stats":server.stats()?}),
    )?;
    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn queued_cancellation_releases_the_session_without_a_gpu_turn() -> Result<()> {
    let (server, [a, b]) = session_pair(1)?;
    let (sender, receiver) = mpsc::channel();
    std::thread::scope(|scope| -> Result<()> {
        let blocker = scope
            .spawn(|| stream_events(&server.address, stream_body(&a, "blocker", 64), "a", sender));
        // Headers arrive only after admission, so this request holds the sole slot.
        receiver.recv_timeout(Duration::from_secs(180))?;
        let waiting = scope.spawn(|| {
            request(
                &server.address,
                "/v1/chat/completions",
                Some(
                    &json!({"session_id":b,"request_id":"waiting","max_tokens":4,
                "messages":[{"role":"user","content":"Hello"}]}),
                ),
            )
        });
        wait_for(
            || Ok(server.stats()?["queued_requests"].as_u64().unwrap() > 0),
            Duration::from_secs(10),
        )?;
        cancel(&server.address, "waiting")?;
        assert_eq!(waiting.join().unwrap()?.status, 499);
        let session = stored_session(&server.address, &b)?;
        assert_eq!(session["messages"], json!([]));
        assert!(session["active_request_id"].is_null());
        cancel(&server.address, "blocker")?;
        blocker.join().unwrap()?;
        Ok(())
    })?;
    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn cancellation_during_prefill_never_commits_a_turn() -> Result<()> {
    let server = Server::start_with_active(900, 2)?;
    let id = session(&server.address, 42)?;
    let mut input = stream_body(&id, "prefill-cancel", 16);
    input["messages"][0]["content"] = json!("A cache keeps recently used data. ".repeat(30));
    let (sender, _receiver) = mpsc::channel();
    let events = std::thread::scope(|scope| -> Result<_> {
        let task = scope.spawn(|| stream_events(&server.address, input, "prefill", sender));
        wait_for(
            || Ok(server.stats()?["current"]["phase"] == "prefill"),
            Duration::from_secs(180),
        )?;
        cancel(&server.address, "prefill-cancel")?;
        task.join().unwrap()
    })?;
    assert_eq!(events.last().unwrap()["usage"]["completion_tokens"], 0);
    assert_eq!(stored_session(&server.address, &id)?["messages"], json!([]));
    assert_eq!(server.stats()?["active_state_reserved_bytes"], 0);
    Ok(())
}
