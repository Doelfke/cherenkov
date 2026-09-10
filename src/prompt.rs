//! Checkpoint-owned Jinja formatting, compiled once when the tokenizer loads.

use anyhow::{Context, Result, ensure};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{Value, json};
use std::{io::ErrorKind as IoErrorKind, path::Path};

pub(crate) struct Prompt {
    pub text: String,
    /// Safe byte prefixes, checked against the complete rendered prompt.
    pub boundaries: [usize; 2],
}

impl Prompt {
    pub(crate) fn raw(text: String) -> Self {
        Self {
            text,
            boundaries: [0, 0],
        }
    }
}

pub(crate) struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub(crate) fn load(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("chat_template.jinja");
        let source = match std::fs::read_to_string(&path) {
            Ok(source) => Some(source),
            Err(error) if error.kind() == IoErrorKind::NotFound => embedded_template(model_dir)?,
            Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
        };

        source.map(Self::new).transpose()
    }

    fn new(source: String) -> Result<Self> {
        let mut env = Environment::new();

        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_recursion_limit(64);
        env.set_fuel(Some(1_000_000));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)
            .context("compiling checkpoint chat template")?;

        Ok(Self { env })
    }

    pub(crate) fn user(&self, content: &str) -> Result<Prompt> {
        self.chat(&[json!({"role": "user", "content": content})])
    }

    pub(crate) fn chat(&self, messages: &[Value]) -> Result<Prompt> {
        let mut messages = text_messages(messages)?;
        let text = self.render(&messages, true)?;
        let message_end = common_prefix(&text, &self.render(&messages, false)?);
        // Prefix-only renders can be invalid (the template requires a user query).
        // Keep the last user with empty content as a cache probe, and trust only
        // bytes that also occur at the beginning of the full rendered prompt.
        let mut stable_end = 0;

        if let Some(last_user) = messages.iter().rposition(|m| m["role"] == "user") {
            messages.truncate(last_user + 1);

            messages[last_user]["content"] = json!("");

            if let Ok(prefix) = self.render(&messages, false) {
                stable_end = common_prefix(&text, &prefix);
            }
        }

        Ok(Prompt {
            text,
            boundaries: [stable_end, message_end],
        })
    }

    fn render(&self, messages: &[Value], add_generation_prompt: bool) -> Result<String> {
        // Preserve the engine's direct-answer default. Other formatting and
        // reasoning-history defaults come from the checkpoint's template.
        self.render_context(&json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": false,
        }))
    }

    fn render_context(&self, context: &Value) -> Result<String> {
        self.env
            .get_template("chat")?
            .render(context)
            .map_err(|error| anyhow::anyhow!("chat template: {error}"))
    }
}

fn embedded_template(model_dir: &Path) -> Result<Option<String>> {
    let path = model_dir.join("tokenizer_config.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let config: Value =
        serde_json::from_slice(&bytes).context("reading tokenizer configuration")?;
    let template = &config["chat_template"];

    if let Some(source) = template.as_str() {
        return Ok(Some(source.to_owned()));
    }

    // Transformers also saves named templates; chat without tools uses "default".
    Ok(template.as_array().and_then(|templates| {
        templates
            .iter()
            .find(|t| t["name"] == "default")?
            .get("template")?
            .as_str()
            .map(str::to_owned)
    }))
}

fn text_messages(messages: &[Value]) -> Result<Vec<Value>> {
    ensure!(!messages.is_empty(), "messages must not be empty");

    let mut messages = messages.to_vec();

    for message in &mut messages {
        let role = message["role"]
            .as_str()
            .context("message role must be a string")?;

        ensure!(
            ["system", "developer", "user", "assistant"].contains(&role),
            "unsupported role {role}"
        );
        ensure!(
            message["content"].is_string(),
            "message content must be text"
        );
        ensure!(
            message["tool_calls"].is_null(),
            "tool calls are not supported"
        );

        if role == "developer" {
            message["role"] = json!("system");
        }
    }

    Ok(messages)
}

fn common_prefix(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(a, b)| a == b)
        .map(|(c, _)| c.len_utf8())
        .sum()
}

#[cfg(test)]
pub(crate) fn fixture_template() -> ChatTemplate {
    ChatTemplate::new(include_str!("../tests/fixtures/prompt/chat_template.jinja").to_owned())
        .expect("checkpoint template fixture")
}

#[cfg(test)]
#[path = "../tests/unit/prompt.rs"]
mod tests;
