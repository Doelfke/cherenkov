use super::*;

#[test]
#[ignore = "requires the real model and Metal"]
fn request_capacity_rejects_overflow_and_reuses_cancelled_ids() -> Result<()> {
    let server = Server::configured(ServerSettings {
        queued_requests: 1,
        ..ServerSettings::default()
    })?;
    let a = session(&server.address, 42)?;
    let b = session(&server.address, 71)?;
    let (sender, receiver) = mpsc::channel();

    std::thread::scope(|scope| -> Result<()> {
        let active = scope.spawn(|| {
            stream_events(
                &server.address,
                stream_body(&a, "capacity-active", 64),
                "a",
                sender,
            )
        });

        receiver.recv_timeout(Duration::from_secs(180))?;
        assert_eq!(
            request(
                &server.address,
                "/v1/chat/completions",
                Some(&completion_body(&a, "Busy", 1))
            )?
            .status,
            409
        );

        let waiting = scope.spawn(|| {
            let mut body = completion_body(&b, "Hello", 4);
            body["request_id"] = json!("capacity-waiting");

            request(&server.address, "/v1/chat/completions", Some(&body))
        });

        wait_for(
            || Ok(server.stats()?["queued_requests"] == 1),
            Duration::from_secs(10),
        )?;
        assert_capacity_rejections(&server, &a)?;
        cancel(&server.address, "capacity-waiting")?;
        assert_eq!(waiting.join().unwrap()?.status, 499);
        cancel(&server.address, "capacity-active")?;
        active.join().unwrap()?;

        Ok(())
    })?;
    wait_idle(&server)?;

    let mut retry = completion_body(&a, "Hello", 1);
    retry["request_id"] = json!("capacity-active");
    let response = request(&server.address, "/v1/chat/completions", Some(&retry))?.success()?;

    assert_eq!(response["id"], "capacity-active");
    assert_eq!(response["usage"]["completion_tokens"], 1);
    wait_idle(&server)?;

    Ok(())
}

fn assert_capacity_rejections(server: &Server, session: &str) -> Result<()> {
    for (id, status) in [("capacity-active", 409), ("overflow", 503)] {
        let mut body = completion_body(session, "Hello", 1);
        body["request_id"] = json!(id);
        let response = request(&server.address, "/v1/chat/completions", Some(&body))?;

        assert_eq!(response.status, status, "{}", response.body);
    }

    assert_eq!(server.stats()?["rejected_requests"], 1);

    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn memory_admission_waits_then_reuses_a_cancelled_requests_reservation() -> Result<()> {
    let server = Server::configured(ServerSettings {
        active_requests: 2,
        active_state_mib: 300,
        ..ServerSettings::default()
    })?;
    let a = session(&server.address, 42)?;
    let b = session(&server.address, 71)?;
    let (sender, receiver) = mpsc::channel();

    std::thread::scope(|scope| -> Result<()> {
        let running = scope.spawn(|| {
            stream_events(
                &server.address,
                stream_body(&a, "memory-active", 64),
                "a",
                sender,
            )
        });

        receiver.recv_timeout(Duration::from_secs(180))?;

        let waiting = scope.spawn(|| {
            request(
                &server.address,
                "/v1/chat/completions",
                Some(&completion_body(&b, "Explain cache locality.", 4)),
            )
        });

        wait_for(
            || {
                let stats = server.stats()?;

                Ok(stats["queued_requests"] == 1 && stats["active_state_reserved_bytes"] != 0)
            },
            Duration::from_secs(10),
        )?;

        let stats = server.stats()?;

        assert_eq!(stats["active_requests"], 1);

        let reserved = stats["active_state_reserved_bytes"]
            .as_u64()
            .context("reservation")?;

        assert!(
            reserved > 150 * 1024 * 1024 && reserved <= 300 * 1024 * 1024,
            "{stats}"
        );
        cancel(&server.address, "memory-active")?;
        running.join().unwrap()?;
        assert_eq!(
            waiting.join().unwrap()?.success()?["usage"]["completion_tokens"],
            4
        );

        Ok(())
    })?;
    wait_idle(&server)?;
    assert_eq!(stored_session(&server.address, &a)?["messages"], json!([]));
    assert_eq!(
        stored_session(&server.address, &b)?["messages"]
            .as_array()
            .context("history")?
            .len(),
        2
    );

    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn an_oversized_request_is_rejected_before_gpu_work_and_unpins_the_session() -> Result<()> {
    let server = Server::configured(ServerSettings {
        active_state_mib: 1,
        ..ServerSettings::default()
    })?;

    failed_turn(&server, "Hello", 4, 503, "active_state_mib")?;

    let stats = server.stats()?;

    assert_eq!(stats["prompt_tokens"], 0);
    assert_eq!(stats["generated_tokens"], 0);
    assert_eq!(stats["rejected_requests"], 1);

    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn response_limit_failure_rolls_back_and_the_session_can_retry() -> Result<()> {
    let server = Server::configured(ServerSettings {
        response_bytes: 16,
        ..ServerSettings::default()
    })?;
    let id = failed_turn(
        &server,
        "Explain how hash tables work in detail.",
        64,
        500,
        "response_bytes",
    )?;
    let retry = request(
        &server.address,
        "/v1/chat/completions",
        Some(&completion_body(
            &id,
            "Explain how hash tables work in detail.",
            1,
        )),
    )?
    .success()?;

    assert_eq!(retry["usage"]["completion_tokens"], 1);
    wait_idle(&server)?;
    assert_eq!(server.stats()?["failed_requests"], 1);
    assert_eq!(server.stats()?["completed_requests"], 1);

    Ok(())
}

fn failed_turn(
    server: &Server,
    prompt: &str,
    tokens: usize,
    status: u16,
    reason: &str,
) -> Result<String> {
    let id = session(&server.address, 42)?;
    let response = request(
        &server.address,
        "/v1/chat/completions",
        Some(&completion_body(&id, prompt, tokens)),
    )?;

    assert_eq!(response.status, status);
    assert!(response.body.contains(reason), "{}", response.body);
    wait_idle(server)?;

    let session = stored_session(&server.address, &id)?;

    assert_eq!(session["messages"], json!([]));
    assert!(session["active_request_id"].is_null());

    Ok(id)
}
