use super::*;
use crate::{
    config::Config,
    server::{
        registry::Registry,
        tests::fixtures::{begin_turn, message, new_session},
    },
};
use serde_json::json;

struct PreparedOutput {
    store: Arc<std::sync::Mutex<crate::server::sessions::Store>>,
    id: String,
    ticket: Arc<Ticket>,
    turn: Commit,
}

fn prepared_output(request: &str) -> PreparedOutput {
    let (store, id) = new_session(&Config::default(), json!({"seed":42}));
    let registry = Arc::new(Registry::default());
    let ticket = registry.register(Some(request), 1).expect("test request");
    let body = message(&id, "Hello");
    let turn = begin_turn(&store, &body, &ticket.id)
        .prepare("Hi", None, None)
        .expect("prepared turn");

    PreparedOutput {
        store,
        id,
        ticket,
        turn,
    }
}

fn terminal(turn: Commit) -> Frame {
    Frame::Finish {
        text: "Hi".into(),
        reason: "length",
        usage: json!({"completion_tokens":1}),
        tool_calls: None,
        turn: Some(Box::new(turn)),
    }
}

// Exercise the real writer boundary with a prepared turn, without a GPU or
// timing-sensitive sockets. Streaming text is provisional until Finish wins.
#[test]
fn cancellation_before_final_frame_rolls_back_the_prepared_turn() {
    for (cancelled, reason, messages) in [(true, "cancelled", 0), (false, "length", 2)] {
        let PreparedOutput {
            store,
            id,
            ticket,
            turn,
        } = prepared_output("writer-boundary");
        let (sender, receiver) = mpsc::channel();

        sender.send(Frame::Text("Hi".into())).unwrap();
        sender.send(terminal(turn)).unwrap();

        if cancelled {
            ticket.cancel();
        }

        let mut bytes = Vec::new();
        let mut response =
            Response::for_writer(&mut bytes, ApiKind::Chat, &ticket.id, 0, true, true);

        assert_eq!(
            write_frames(&mut response, receiver, &ticket).unwrap(),
            !cancelled
        );

        let session = store.lock().unwrap().show(&id).unwrap();

        assert_eq!(session["messages"].as_array().unwrap().len(), messages);
        assert!(session["active_request_id"].is_null());

        let output = String::from_utf8(bytes).unwrap();

        assert!(output.contains(&format!("\"finish_reason\":\"{reason}\"")));
        assert!(output.ends_with("data: [DONE]\n\n"));

        if !cancelled {
            ticket.cancel();
            assert!(!ticket.cancelled());
        }
    }
}

#[test]
fn a_full_output_queue_cancels_without_publishing_or_blocking() {
    let PreparedOutput {
        store,
        id,
        ticket,
        turn,
    } = prepared_output("slow-client");
    let (sender, _receiver) = mpsc::sync_channel(1);
    let output = Output {
        sender,
        ticket: ticket.clone(),
    };

    output.send(Frame::Text("Hi".into())).unwrap();
    assert!(output.send(terminal(turn)).is_err());
    assert!(ticket.cancelled());

    let session = store.lock().unwrap().show(&id).unwrap();

    assert_eq!(session["messages"], json!([]));
    assert!(session["active_request_id"].is_null());
}

struct BrokenWriter;

impl std::io::Write for BrokenWriter {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn writer_failure_respects_the_publication_boundary() {
    // Streaming writes headers before publication. A non-streaming response's
    // first write follows publication, so a late disconnect retains that turn.
    for (streaming, expected_messages) in [(true, 0), (false, 2)] {
        let PreparedOutput {
            store,
            id,
            ticket,
            turn,
        } = prepared_output("disconnected");
        let (sender, receiver) = mpsc::channel();

        sender.send(terminal(turn)).unwrap();

        let mut writer = BrokenWriter;
        let mut response =
            Response::for_writer(&mut writer, ApiKind::Chat, &ticket.id, 0, streaming, true);

        assert!(write_frames(&mut response, receiver, &ticket).is_err());

        let session = store.lock().unwrap().show(&id).unwrap();

        assert_eq!(
            session["messages"].as_array().unwrap().len(),
            expected_messages
        );
        assert!(session["active_request_id"].is_null());
        ticket.cancel();
        assert_eq!(ticket.cancelled(), streaming);
    }
}
