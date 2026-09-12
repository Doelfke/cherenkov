use super::*;
use crate::server::tests::fixtures::{begin_turn, message, new_session};

fn usage() -> UsageStats {
    UsageStats {
        prompt_tokens: 20,
        cached_tokens: 8,
        generated_tokens: 5,
        prefill_seconds: 0.25,
        decode_seconds: 0.5,
    }
}

#[test]
fn session_usage_commits_with_history_and_accumulates_across_turns() {
    let (store, id) = new_session(&Config::default(), json!({}));

    for index in 1..=2 {
        let turn = begin_turn(&store, &message(&id, "Hello"), &format!("req-{index}"));
        let commit = turn.prepare("Hi", None, usage(), None).unwrap();
        let before = store.lock().unwrap().show(&id).unwrap();

        assert_eq!(before["stats"]["committed_turns"], index - 1);
        assert!(before["stats"]["reserved_history_bytes"].as_u64().unwrap() > 0);
        commit.publish();

        let after = store.lock().unwrap().show(&id).unwrap();
        let stats = &after["stats"];

        assert_eq!(stats["committed_turns"], index);
        assert_eq!(stats["usage"]["prompt_tokens"], 20 * index);
        assert_eq!(stats["usage"]["cached_tokens"], 8 * index);
        assert_eq!(stats["usage"]["generated_tokens"], 5 * index);
        assert_eq!(stats["usage"]["prefill_seconds"], 0.25 * f64::from(index));
        assert_eq!(stats["usage"]["decode_seconds"], 0.5 * f64::from(index));
        assert_eq!(stats["reserved_history_bytes"], 0);
        assert_eq!(
            stats["history_bytes"],
            serde_json::to_vec(&after["messages"]).unwrap().len()
        );
    }
}

#[test]
fn unpublished_usage_does_not_leak_to_this_or_another_session() {
    let config = Config::default();
    let (store, id) = new_session(&config, json!({}));
    let other = store.lock().unwrap().create(&json!({}), &config).unwrap();
    let turn = begin_turn(&store, &message(&id, "Hello"), "cancelled");
    let commit = turn.prepare("unfinished", None, usage(), None).unwrap();

    drop(commit);

    for id in [&id, &other] {
        let session = store.lock().unwrap().show(id).unwrap();

        assert_eq!(session["stats"]["committed_turns"], 0);
        assert_eq!(session["stats"]["usage"]["generated_tokens"], 0);
        assert_eq!(session["stats"]["reserved_history_bytes"], 0);
        assert_eq!(session["stats"]["history_bytes"], 2);
    }
}

#[test]
fn tool_calls_and_usage_share_the_session_commit_boundary() {
    let (store, id) = new_session(&Config::default(), json!({}));
    let calls = [WireToolCall {
        id: "call-1".to_owned(),
        name: "weather".to_owned(),
        arguments: r#"{"city":"Boston"}"#.to_owned(),
    }];
    let turn = begin_turn(&store, &message(&id, "Weather?"), "first");
    let commit = turn.prepare("", None, usage(), Some(&calls)).unwrap();
    let pending = store.lock().unwrap().show(&id).unwrap();

    assert_eq!(pending["messages"], json!([]));
    assert_eq!(pending["stats"]["committed_turns"], 0);
    commit.publish();

    let committed = store.lock().unwrap().show(&id).unwrap();
    let assistant = &committed["messages"][1];

    assert!(assistant["content"].is_null());
    assert_eq!(assistant["tool_calls"][0]["id"], "call-1");
    assert_eq!(
        assistant["tool_calls"][0]["function"]["arguments"],
        json!({"city": "Boston"})
    );
    assert_eq!(committed["stats"]["usage"]["generated_tokens"], 5);

    let turn = begin_turn(&store, &message(&id, "Again?"), "cancelled");
    let cancelled = turn.prepare("", None, usage(), Some(&calls)).unwrap();

    drop(cancelled);

    let after = store.lock().unwrap().show(&id).unwrap();

    assert_eq!(after["messages"], committed["messages"]);
    assert_eq!(after["stats"]["usage"], committed["stats"]["usage"]);
    assert_eq!(after["stats"]["committed_turns"], 1);
    assert_eq!(after["stats"]["reserved_history_bytes"], 0);
}
