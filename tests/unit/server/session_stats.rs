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
        let commit = turn.prepare("Hi", None, usage()).unwrap();
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
    let commit = turn.prepare("unfinished", None, usage()).unwrap();

    drop(commit);

    for id in [&id, &other] {
        let session = store.lock().unwrap().show(id).unwrap();

        assert_eq!(session["stats"]["committed_turns"], 0);
        assert_eq!(session["stats"]["usage"]["generated_tokens"], 0);
        assert_eq!(session["stats"]["reserved_history_bytes"], 0);
        assert_eq!(session["stats"]["history_bytes"], 2);
    }
}
