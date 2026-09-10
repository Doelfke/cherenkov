//! Endpoint-specific JSON and SSE framing, independent of token generation.

use super::{ApiKind, MODEL, Request, respond, sse};
use anyhow::Result;
use serde_json::{Value, json};
use std::io::Write;

pub(super) struct Response<'a, W: Write> {
    out: &'a mut W,
    kind: ApiKind,
    id: String,
    created: u64,
    streaming: bool,
    include_usage: bool,
}

impl<'a, W: Write> Response<'a, W> {
    pub(super) fn new(
        out: &'a mut W,
        kind: ApiKind,
        serial: u64,
        created: u64,
        request: &Request,
    ) -> Self {
        let prefix = match kind {
            ApiKind::Chat => "chatcmpl",
            ApiKind::Completion => "cmpl",
        };
        Self {
            out,
            kind,
            id: format!("{prefix}-{created}-{serial}"),
            created,
            streaming: request.stream,
            include_usage: request.include_usage,
        }
    }

    pub(super) fn start(&mut self) -> Result<()> {
        if self.streaming {
            self.out.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
            if self.kind == ApiKind::Chat {
                let mut choice = self.delta_choice(Some(""), None);
                choice["delta"]["role"] = json!("assistant");
                self.send_chunk(choice)?;
            }
        }
        Ok(())
    }

    pub(super) fn text(&mut self, text: &str) -> Result<()> {
        if self.streaming {
            self.send_chunk(self.delta_choice(Some(text), None))?;
        }
        Ok(())
    }

    pub(super) fn finish(&mut self, text: &str, finish_reason: &str, usage: Value) -> Result<()> {
        if self.streaming {
            self.send_chunk(self.delta_choice(None, Some(finish_reason)))?;
            if self.include_usage {
                let mut body = self.envelope(vec![]);
                body["usage"] = usage;
                sse(self.out, &body)?;
            }
            self.out.write_all(b"data: [DONE]\n\n")?;
            self.out.flush()?;
        } else {
            let mut choice = self.choice(Some(finish_reason));
            match self.kind {
                ApiKind::Chat => {
                    choice["message"] = json!({"role":"assistant", "content":text});
                }
                ApiKind::Completion => choice["text"] = json!(text),
            }
            let mut body = self.envelope(vec![choice]);
            body["usage"] = usage;
            respond(self.out, 200, &body)?;
        }
        Ok(())
    }

    fn choice(&self, finish_reason: Option<&str>) -> Value {
        let mut choice = json!({"index":0, "finish_reason":finish_reason});
        if self.kind == ApiKind::Completion {
            choice["logprobs"] = Value::Null;
        }
        choice
    }

    // A final chat chunk has an empty delta, not an empty content field.
    fn delta_choice(&self, text: Option<&str>, finish_reason: Option<&str>) -> Value {
        let mut choice = self.choice(finish_reason);
        match self.kind {
            ApiKind::Chat => {
                let mut delta = json!({});
                if let Some(text) = text {
                    delta["content"] = json!(text);
                }
                choice["delta"] = delta;
            }
            ApiKind::Completion => choice["text"] = json!(text.unwrap_or("")),
        }
        choice
    }

    fn envelope(&self, choices: Vec<Value>) -> Value {
        let object = match (self.kind, self.streaming) {
            (ApiKind::Chat, true) => "chat.completion.chunk",
            (ApiKind::Chat, false) => "chat.completion",
            (ApiKind::Completion, _) => "text_completion",
        };
        json!({"id":self.id, "object":object, "created":self.created, "model":MODEL, "choices":choices})
    }

    fn send_chunk(&mut self, choice: Value) -> Result<()> {
        sse(self.out, &self.envelope(vec![choice]))
    }
}
