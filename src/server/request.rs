//! Completion options, chat rendering, tokenization and context policy.

use super::{ApiKind, MODEL, tool_call::ToolCallOutputContract};
use crate::{
    config::Defaults,
    options::Options,
    prompt::{ChatTemplate, Prompt, valid_function_name},
    runner,
    sampling::Sampling,
    tok::ChatTokenizer,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};
use std::sync::Arc;

pub(super) struct Request {
    prompt: Prompt,
    pub(super) max_tokens: usize,
    pub(super) stream: bool,
    pub(super) include_usage: bool,
    no_eos: bool,
    sampling: Sampling,
    context: Option<usize>,
    /// Contract for tool-call output parsing; `Some` while tools are enabled.
    pub(super) tool_contract: Option<Arc<ToolCallOutputContract>>,
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

    // Chat parses tool controls below; completions never take them.
    let tool_keys: &[&str] = match kind {
        ApiKind::Chat => &[],
        ApiKind::Completion => &["tools", "tool_choice", "functions", "function_call"],
    };

    for key in [
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

    for key in tool_keys {
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
    let mut tools = Vec::new();
    let mut tool_contract = None;

    if kind == ApiKind::Chat {
        let parsed = parse_tools(v)?;
        let enabled = parse_tool_choice(v)?;
        parse_parallel_tool_calls(v, enabled && !parsed.is_empty())?;
        parse_legacy_function_controls(v)?;

        if enabled {
            tools = parsed;
            tool_contract = (!tools.is_empty()).then(|| ToolCallOutputContract::from_tools(&tools));
        }
    }

    let prompt = render_prompt(v, kind, session, template, &tools)?;

    Ok(Request {
        prompt,
        max_tokens,
        stream,
        include_usage,
        no_eos: defaults.no_eos,
        sampling,
        context,
        tool_contract,
    })
}

fn render_prompt(
    v: &Value,
    kind: ApiKind,
    session: Option<&SessionInput>,
    template: Option<&ChatTemplate>,
    tools: &[Value],
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
        .chat(messages, Some(tools))
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
/// Shape a request's tools for the checkpoint template; reject types the
/// Qwen4 frontend cannot honor.
fn parse_tools(v: &Value) -> Result<Vec<Value>> {
    let mut tools = Vec::new();

    if v["tools"].is_null() {
        return Ok(tools);
    }

    for entry in v["tools"].as_array().context("tools must be an array")? {
        let ty = entry["type"]
            .as_str()
            .context("tools entries must contain a string type")?;

        ensure!(
            ty == "function",
            "tool type '{ty}' requires a non-function output contract, which is not supported; use function tools"
        );

        let function = entry.get("function").filter(|value| !value.is_null());
        let function = function
            .and_then(Value::as_object)
            .context("function tools must contain a function object")?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .context("function name must be a string")?;

        ensure!(
            valid_function_name(name),
            "function name must match [A-Za-z0-9_-]{{1,64}}"
        );

        let description = function
            .get("description")
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_str()
                    .context("function description must be a string")
            })
            .transpose()?;

        let parameters = function.get("parameters").filter(|value| !value.is_null());
        let parameters = match parameters {
            Some(parameters) => {
                parameters
                    .as_object()
                    .context("function parameters must be a JSON object")?;
                parameters.clone()
            }
            None => json!({"type": "object", "properties": {}}),
        };

        if let Some(strict) = function.get("strict").filter(|value| !value.is_null()) {
            ensure!(strict.is_boolean(), "function strict must be a boolean");

            let strict_value = strict
                .as_bool()
                .context("function strict must be a boolean")?;

            ensure!(
                !strict_value,
                "strict=true requires generated arguments to satisfy the declared schema, which cannot be guaranteed; omit strict or use false"
            );
        }

        // The template emits `tool | tojson`; keep the shape the Qwen tool
        // contract expects: a function wrapper whose non-guaranteed `strict`
        // flag is normalized to false.
        let mut shaped_function = Map::new();
        shaped_function.insert("name".to_owned(), Value::String(name.to_owned()));
        shaped_function.insert("parameters".to_owned(), parameters);
        shaped_function.insert("strict".to_owned(), Value::Bool(false));

        if let Some(description) = description {
            shaped_function.insert(
                "description".to_owned(),
                Value::String(description.to_owned()),
            );
        }

        let mut shaped = Map::new();
        shaped.insert("type".to_owned(), Value::String("function".to_owned()));
        shaped.insert("function".to_owned(), Value::Object(shaped_function));

        tools.push(Value::Object(shaped));
    }

    Ok(tools)
}

/// Resolve `tool_choice`; `false` suppresses the tools block entirely.
fn parse_tool_choice(v: &Value) -> Result<bool> {
    let Some(choice) = v.get("tool_choice").filter(|value| !value.is_null()) else {
        return Ok(true);
    };

    if let Some(ty) = choice.as_str() {
        return match ty {
            "auto" => Ok(true),
            "none" => Ok(false),
            "required" => anyhow::bail!(
                "tool_choice='required' requires at least one tool call, which cannot be guaranteed; use 'auto' or 'none'"
            ),
            other => anyhow::bail!(
                "tool_choice must be 'auto', 'none', or a function choice, not '{other}'"
            ),
        };
    }

    let object = choice
        .as_object()
        .context("tool_choice must be a string or object")?;
    let ty = object["type"]
        .as_str()
        .context("tool_choice objects must contain a string type")?;

    if ty == "function" {
        anyhow::bail!(
            "tool_choice for a specific function forces that function to be called, which cannot be guaranteed; use 'auto' or 'none'"
        );
    }

    anyhow::bail!("unsupported tool_choice type '{ty}'")
}

fn parse_parallel_tool_calls(v: &Value, tools_enabled: bool) -> Result<()> {
    let Some(value) = v
        .get("parallel_tool_calls")
        .filter(|value| !value.is_null())
    else {
        return Ok(());
    };

    let value = value
        .as_bool()
        .context("parallel_tool_calls must be a boolean")?;

    ensure!(
        value || !tools_enabled,
        "parallel_tool_calls=false requires the model to emit at most one tool call, which cannot be guaranteed while tools are enabled"
    );

    Ok(())
}

/// Tolerate legacy OpenAI function controls the Qwen4 frontend ignores.
fn parse_legacy_function_controls(v: &Value) -> Result<()> {
    let Some(functions) = v.get("functions").filter(|value| !value.is_null()) else {
        return legacy_function_call(v);
    };

    let entries = functions.as_array().context("functions must be an array")?;

    ensure!(
        entries.is_empty(),
        "non-empty legacy functions require the single-function-call response contract, which is not supported; use tools instead"
    );

    legacy_function_call(v)
}

fn legacy_function_call(v: &Value) -> Result<()> {
    let Some(choice) = v.get("function_call").filter(|value| !value.is_null()) else {
        return Ok(());
    };

    if let Some(value) = choice.as_str() {
        ensure!(
            value == "none" || value == "auto",
            "function_call must be 'none', 'auto', or a named function choice"
        );

        return Ok(());
    }

    ensure!(
        choice.is_object(),
        "function_call must be 'none', 'auto', or an object"
    );
    anyhow::bail!(
        "a named legacy function_call forces that function to be called, which cannot be guaranteed; use tool_choice instead"
    );
}
#[cfg(test)]
#[path = "../../tests/unit/server/request.rs"]
mod tests;
