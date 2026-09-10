//! Local OpenAI-compatible HTTP adapter. One thread owns the loaded GPU;
//! connection readers feed a bounded queue, so parked sockets do not block it.

use crate::{
    config::{Defaults, Source},
    control::{self, State, state::Versioned},
    options::Options,
    prefix_cache::PrefixCache,
    qwen4_exp, runner,
    tok::ChatTokenizer,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

mod response;
mod worker;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ApiKind {
    Chat,
    Completion,
}

const MODEL: &str = "cherenkov";

struct Job {
    stream: TcpStream,
    body: Value,
    kind: ApiKind,
    settings: Versioned,
}

struct Request {
    prompt: String,
    stable_prefix: String,
    message_prefix: String,
    max_tokens: usize,
    stream: bool,
    include_usage: bool,
    no_eos: bool,
}

/// Tokenization and policy checks finish before the response starts.
struct PreparedRequest {
    request: Request,
    ids: Vec<u32>,
    boundaries: Vec<usize>,
    options: Options,
}

impl Request {
    fn prepare(
        self,
        tok: &ChatTokenizer,
        options: &Options,
        max_output_tokens: usize,
    ) -> Result<PreparedRequest> {
        ensure!(
            self.max_tokens <= max_output_tokens,
            "max_tokens exceeds server output policy"
        );
        let ids = tok.encode(&self.prompt)?;
        runner::check_budget(
            ids.len(),
            self.max_tokens,
            options.drafts as usize,
            options.max_ctx,
        )?;
        let mut boundaries = Vec::new();
        for text in [&self.stable_prefix, &self.message_prefix] {
            let prefix = tok.encode(text)?;
            boundaries.push(ids.iter().zip(&prefix).take_while(|(a, b)| a == b).count());
        }
        let mut options = options.clone();
        options.no_eos = self.no_eos;
        Ok(PreparedRequest {
            request: self,
            ids,
            boundaries,
            options,
        })
    }
}

pub fn serve(source: Source) -> Result<()> {
    let config = source.resolve()?;
    let model_dir = config.model_dir()?;
    let mut options = config.options();
    options.repack = source.overrides.repack;
    ensure!(
        !options.repack || options.build_missing_store,
        "--repack conflicts with build_missing_store=false"
    );
    options.validate()?;
    let cache_bytes = config.cache_bytes();
    let cache = PrefixCache::new(
        cache_bytes,
        config.limits.cache_max_entries,
        config.limits.cache_idle_seconds,
    );
    let state = Arc::new(State::new(source, config.clone()));
    let _control = control::Listener::start(&config.server.socket, state.clone())?;
    eprintln!("control socket: {}", config.server.socket.display());
    let listener =
        TcpListener::bind(("127.0.0.1", config.server.port)).context("binding server")?;
    let address = listener.local_addr()?;
    let tok = ChatTokenizer::load(&model_dir)?;
    let packed = qwen4_exp::packed::Packed::open(&model_dir)?;
    let gpu = qwen4_exp::gpu::Gpu::load_bounded(
        &packed,
        options.max_ctx,
        &options,
        cache_bytes,
        Some(config.memory_bytes()),
    )?;
    ensure!(
        gpu.allocated_gb() + cache_bytes as f64 / 1e9 <= config.limits.memory_gb,
        "Metal plus prefix cache exceeds configured memory budget"
    );
    state.observe(&gpu, &cache);
    state.update(|s| {
        s.ready = true;
        s.http_address = Some(address.to_string());
    });
    if options.cut_weak > 0.0 {
        eprintln!("WARNING: --cut-weak makes output depend on disk timing and non-reproducible.");
    }
    eprintln!(
        "cherenkov serving http://{address}/v1, model {MODEL}, {:.2} GB Metal",
        gpu.allocated_gb()
    );
    let (tx, rx) = mpsc::sync_channel::<Job>(config.limits.queued_requests);
    let connections = state.clone();
    std::thread::spawn(move || {
        let active = Arc::new(AtomicUsize::new(0));
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            if active.fetch_add(1, Ordering::AcqRel) >= config.limits.http_readers {
                active.fetch_sub(1, Ordering::AcqRel);
                stream.set_write_timeout(Some(Duration::from_secs(1))).ok();
                connections.update(|s| s.rejected_requests += 1);
                let _ = error(&mut stream, 503, "too many connections");
                continue;
            }
            let tx = tx.clone();
            let active = active.clone();
            let state = connections.clone();
            std::thread::spawn(move || {
                if let Err(e) = connection(stream, &tx, &state) {
                    eprintln!("HTTP connection: {e:#}");
                }
                active.fetch_sub(1, Ordering::AcqRel);
            });
        }
    });
    let mut worker = worker::Worker::new(gpu, tok, options, cache, state);
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(job) => worker.handle(job),
            Err(mpsc::RecvTimeoutError::Timeout) => worker.expire_cache(),
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

fn connection(mut stream: TcpStream, queue: &mpsc::SyncSender<Job>, state: &State) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    stream.set_nodelay(true)?;
    let settings = state.config();
    let (method, path, body) = match read_http(&mut stream, settings.config.limits.request_bytes) {
        Ok(r) => r,
        Err(e) => return error(&mut stream, 400, &e.to_string()),
    };
    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, &json!({"status":"ok"})),
        ("GET", "/v1/models") => respond(
            &mut stream,
            200,
            &json!({
                "object":"list", "data":[{"id":MODEL,"object":"model","created":0,"owned_by":"local"}]
            }),
        ),
        ("POST", "/v1/chat/completions" | "/v1/completions") => {
            let body = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => return error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            let job = Job {
                stream,
                body,
                settings,
                kind: if path == "/v1/chat/completions" {
                    ApiKind::Chat
                } else {
                    ApiKind::Completion
                },
            };
            state.update(|s| s.queued_requests += 1);
            match queue.try_send(job) {
                Ok(()) => Ok(()),
                Err(mpsc::TrySendError::Full(mut job)) => {
                    state.update(|s| {
                        s.queued_requests -= 1;
                        s.rejected_requests += 1;
                    });
                    error(&mut job.stream, 503, "generation queue is full")
                }
                Err(mpsc::TrySendError::Disconnected(mut job)) => {
                    state.update(|s| {
                        s.queued_requests -= 1;
                        s.rejected_requests += 1;
                    });
                    error(&mut job.stream, 503, "model unavailable")
                }
            }
        }
        _ => error(&mut stream, 404, "unknown endpoint"),
    }
}

fn line(reader: &mut impl BufRead, remaining: &mut usize) -> Result<String> {
    let mut bytes = Vec::new();
    let n = reader
        .take((*remaining + 1) as u64)
        .read_until(b'\n', &mut bytes)?;
    ensure!(
        n > 0 && n <= *remaining && bytes.ends_with(b"\n"),
        "invalid or oversized HTTP headers"
    );
    *remaining -= n;
    Ok(String::from_utf8(bytes)?
        .trim_end_matches(['\r', '\n'])
        .to_owned())
}

fn read_http(stream: &mut TcpStream, max_body: usize) -> Result<(String, String, Vec<u8>)> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut remaining = 16 * 1024;
    let first = line(&mut reader, &mut remaining)?;
    let parts: Vec<_> = first.split_whitespace().collect();
    ensure!(
        parts.len() == 3 && parts[2].starts_with("HTTP/1."),
        "invalid request line"
    );
    let mut length = None;
    let mut expect = false;
    loop {
        let h = line(&mut reader, &mut remaining)?;
        if h.is_empty() {
            break;
        }
        let (key, value) = h.split_once(':').context("invalid header")?;
        match key.to_ascii_lowercase().as_str() {
            "content-length" => {
                ensure!(length.is_none(), "duplicate Content-Length");
                length = Some(
                    value
                        .trim()
                        .parse::<usize>()
                        .context("invalid Content-Length")?,
                );
            }
            "transfer-encoding" => {
                anyhow::bail!("Transfer-Encoding is unsupported; send Content-Length")
            }
            "expect" => {
                ensure!(
                    value.trim().eq_ignore_ascii_case("100-continue"),
                    "unsupported Expect"
                );
                expect = true;
            }
            _ => {}
        }
    }
    let length = length.unwrap_or(0);
    ensure!(
        length <= max_body,
        "request body exceeds server limit of {max_body} bytes"
    );
    if expect {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        stream.flush()?;
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok((parts[0].to_owned(), parts[1].to_owned(), body))
}

fn parse_request(v: &Value, kind: ApiKind, defaults: &Defaults) -> Result<Request> {
    ensure!(v.is_object(), "request must be a JSON object");
    ensure!(
        v["model"].is_null() || v["model"].as_str() == Some(MODEL),
        "model must be cherenkov"
    );
    for (key, allowed) in [
        ("temperature", 0.0),
        ("top_p", 1.0),
        ("n", 1.0),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
    ] {
        ensure!(
            v[key].is_null() || v[key].as_f64() == Some(allowed),
            "{key} must be {allowed}; this engine uses greedy decoding"
        );
    }
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
    let mut stable_prefix = String::new();
    let mut message_prefix = String::new();
    let prompt = if kind == ApiKind::Chat {
        let messages = v["messages"]
            .as_array()
            .context("messages must be an array")?;
        ensure!(!messages.is_empty(), "messages must not be empty");
        let mut prompt = String::new();
        for message in messages {
            let role = message["role"]
                .as_str()
                .context("message role must be a string")?;
            ensure!(
                ["system", "developer", "user", "assistant"].contains(&role),
                "unsupported role {role}"
            );
            ensure!(
                message["tool_calls"].is_null(),
                "tool calls are not supported"
            );
            if role == "user" {
                stable_prefix = prompt.clone();
            }
            let role = if role == "developer" { "system" } else { role };
            let content = message["content"]
                .as_str()
                .context("message content must be text")?;
            prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        message_prefix = prompt.clone();
        prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        prompt
    } else {
        v["prompt"]
            .as_str()
            .context("prompt must be a string")?
            .to_owned()
    };
    Ok(Request {
        prompt,
        stable_prefix,
        message_prefix,
        max_tokens,
        stream,
        include_usage,
        no_eos: defaults.no_eos,
    })
}

fn respond(out: &mut impl Write, status: u16, body: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(body)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(
        out,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    )?;
    out.write_all(&bytes)?;
    out.flush()?;
    Ok(())
}

fn error(out: &mut TcpStream, status: u16, message: &str) -> Result<()> {
    respond(
        out,
        status,
        &json!({"error":{"message":message,"type":if status >= 500 {"server_error"} else {"invalid_request_error"},"param":null,"code":null}}),
    )
}

fn sse(out: &mut impl Write, body: &Value) -> Result<()> {
    writeln!(out, "data: {}\n", serde_json::to_string(body)?)?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/server.rs"]
mod tests;
