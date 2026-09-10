//! Completion options, chat rendering, tokenization and context policy.

use super::{ApiKind, MODEL};
use crate::{
    config::Defaults,
    options::Options,
    prompt::{ChatTemplate, Prompt},
    runner,
    sampling::Sampling,
    tok::ChatTokenizer,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

pub(super) struct Request {
    prompt: Prompt,
    pub(super) max_tokens: usize,
    pub(super) stream: bool,
    pub(super) include_usage: bool,
    no_eos: bool,
    sampling: Sampling,
    context: Option<usize>,
}

/// Resolved session input; the original HTTP JSON remains unchanged.
pub(super) struct SessionInput {
    pub messages: Vec<Value>,
    pub sampling: Sampling,
    pub context: usize,
}

/// Tokenization and policy checks finish before the response starts.
pub(super) struct PreparedRequest {
    pub(super) request: Request,
    pub(super) ids: Vec<u32>,
    pub(super) boundaries: Vec<usize>,
    pub(super) options: Options,
}

impl Request {
    pub(super) fn prepare(
        self,
        tok: &ChatTokenizer,
        options: &Options,
        max_output_tokens: usize,
    ) -> Result<PreparedRequest> {
        ensure!(
            self.max_tokens <= max_output_tokens,
            "max_tokens exceeds server output policy"
        );

        let ids = tok.encode(&self.prompt.text)?;
        let mut options = options.clone();
        options.no_eos = self.no_eos;
        options.sampling = self.sampling.clone();

        if let Some(context) = self.context {
            ensure!(
                context > 0 && context <= options.max_ctx,
                "context_tokens exceeds server capacity"
            );

            options.max_ctx = context;
        }

        runner::check_budget(
            ids.len(),
            self.max_tokens,
            options.effective_drafts(),
            options.max_ctx,
        )?;

        let mut boundaries = Vec::new();

        for (index, end) in self.prompt.boundaries.into_iter().enumerate() {
            let prefix = tok.encode(&self.prompt.text[..end])?;
            let mut matched = ids.iter().zip(&prefix).take_while(|(a, b)| a == b).count();

            // MTP state depends on the following token. At the reusable chat
            // prefix, keep that token inside the known-stable rendered text.
            if index == 0 && options.effective_drafts() > 0 {
                matched = matched.saturating_sub(1);
            }

            boundaries.push(matched);
        }

        Ok(PreparedRequest {
            request: self,
            ids,
            boundaries,
            options,
        })
    }
}

pub(super) fn parse_request(
    v: &Value,
    kind: ApiKind,
    defaults: &Defaults,
    session: Option<&SessionInput>,
    template: Option<&ChatTemplate>,
) -> Result<Request> {
    ensure!(v.is_object(), "request must be a JSON object");
    ensure!(
        v["model"].is_null() || v["model"].as_str() == Some(MODEL),
        "model must be cherenkov"
    );
    ensure!(
        v["n"].is_null() || v["n"].as_u64() == Some(1),
        "n must be 1"
    );

    let (sampling, context) = match session {
        Some(input) => (input.sampling.clone(), Some(input.context)),
        None => (
            Sampling::from_request(v, &defaults.sampling)?,
            context_tokens(v)?,
        ),
    };

    for key in [
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "stop",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "suffix",
        "reasoning_effort",
        "chat_template_kwargs",
        "previous_response_id",
        "conversation",
        "min_p",
        "repetition_penalty",
    ] {
        ensure!(v[key].is_null(), "{key} is not supported");
    }

    ensure!(
        v["response_format"].is_null() || v["response_format"] == json!({"type":"text"}),
        "only text response_format is supported"
    );

    let max_tokens = v
        .get("max_completion_tokens")
        .filter(|v| !v.is_null())
        .or_else(|| v.get("max_tokens").filter(|v| !v.is_null()))
        .map(|n| {
            n.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n > 0)
                .context("max_tokens must be a positive integer")
        })
        .transpose()?
        .unwrap_or(defaults.max_tokens);
    let stream = v
        .get("stream")
        .map(|v| v.as_bool().context("stream must be boolean"))
        .transpose()?
        .unwrap_or(defaults.stream);
    let include_usage = match v.get("stream_options").filter(|v| !v.is_null()) {
        Some(o) => {
            ensure!(o.is_object(), "stream_options must be an object");

            o.get("include_usage")
                .map(|v| v.as_bool().context("include_usage must be boolean"))
                .transpose()?
                .unwrap_or(defaults.include_usage)
        }
        None => defaults.include_usage,
    };
    let prompt = render_prompt(v, kind, session, template)?;

    Ok(Request {
        prompt,
        max_tokens,
        stream,
        include_usage,
        no_eos: defaults.no_eos,
        sampling,
        context,
    })
}

fn render_prompt(
    v: &Value,
    kind: ApiKind,
    session: Option<&SessionInput>,
    template: Option<&ChatTemplate>,
) -> Result<Prompt> {
    if kind == ApiKind::Completion {
        let text = v["prompt"].as_str().context("prompt must be a string")?;

        return Ok(Prompt::raw(text.to_owned()));
    }

    let messages = match session {
        Some(input) => input.messages.as_slice(),
        None => v["messages"]
            .as_array()
            .context("messages must be an array")?,
    };

    template
        .context("checkpoint has no chat template")?
        .chat(messages)
}

fn context_tokens(v: &Value) -> Result<Option<usize>> {
    v.get("context_tokens")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .context("context_tokens must be a positive integer")
        })
        .transpose()
}

#[cfg(test)]
#[path = "../../tests/unit/server/request.rs"]
mod tests;
