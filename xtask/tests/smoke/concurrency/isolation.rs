use super::*;

#[test]
#[ignore = "requires the real model and Metal"]
fn disconnected_stream_releases_state_and_does_not_stop_another_session() -> Result<()> {
    let (server, [a, b]) = session_pair(2)?;
    let mut abandoned = EventStream::open(&server.address, &stream_body(&a, "abandoned", 64))?;

    // Read a generated delta so the disconnect interrupts a live decode.
    while let Some(event) = abandoned.next()? {
        if event["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
        {
            break;
        }
    }

    abandoned.disconnect()?;

    let response = request(
        &server.address,
        "/v1/chat/completions",
        Some(&completion_body(&b, "Explain a linked list.", 4)),
    )?
    .success()?;

    assert_eq!(response["usage"]["completion_tokens"], 4);
    wait_idle(&server)?;
    assert_eq!(stored_session(&server.address, &a)?["messages"], json!([]));
    assert_eq!(server.stats()?["cancelled_requests"], 1);

    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn four_greedy_and_sampled_sessions_progress_and_commit_independently() -> Result<()> {
    let server = Server::configured(ServerSettings {
        active_requests: 4,
        active_state_mib: 1536,
        ..ServerSettings::default()
    })?;
    let ids: Vec<_> = (0..4)
        .map(|i| session(&server.address, 42 + i))
        .collect::<Result<_>>()?;
    let prompts = [
        "Explain an oak tree.",
        "Explain a hash table.",
        "Explain ocean tides.",
        "Explain a bicycle.",
    ];
    let (sender, receiver) = mpsc::channel();
    let events = std::thread::scope(|scope| -> Result<Vec<_>> {
        let tasks: Vec<_> = ids
            .iter()
            .zip(prompts)
            .enumerate()
            .map(|(i, (id, prompt))| {
                let mut body = stream_body(id, &format!("four-{i}"), 12);
                body["messages"][0]["content"] = json!(prompt);
                body["temperature"] = json!(if i % 2 == 0 { 0.0 } else { 0.7 });
                let sender = sender.clone();
                let address = &server.address;

                scope.spawn(move || stream_events(address, body, ["a", "b", "c", "d"][i], sender))
            })
            .collect();

        wait_for_four_streams(&receiver)?;

        tasks.into_iter().map(|t| t.join().unwrap()).collect()
    })?;

    wait_idle(&server)?;

    for ((id, prompt), events) in ids.iter().zip(prompts).zip(&events) {
        assert_eq!(
            events.last().context("usage event")?["usage"]["completion_tokens"],
            12
        );

        let text: String = events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        let history = stored_session(&server.address, id)?;

        assert_eq!(
            history["messages"],
            json!([
                {"role":"user","content":prompt}, {"role":"assistant","content":text}
            ])
        );
    }

    assert_eq!(server.stats()?["completed_requests"], 4);
    save_result(
        "sessions-four",
        json!({"stats":server.stats()?, "streams":events}),
    )?;

    Ok(())
}

fn wait_for_four_streams(receiver: &mpsc::Receiver<(&str, Value)>) -> Result<()> {
    let mut progressed = std::collections::HashSet::new();

    while progressed.len() < 4 {
        let (id, event) = receiver.recv_timeout(Duration::from_secs(180))?;

        anyhow::ensure!(event.get("error").is_none(), "{event}");
        anyhow::ensure!(
            event["choices"][0]["finish_reason"].is_null(),
            "a request finished before all four progressed"
        );

        if event["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
        {
            progressed.insert(id.to_owned());
        }
    }

    Ok(())
}
