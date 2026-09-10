use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use xtask::{capture::ChildGuard, util};

pub fn binary() -> PathBuf {
    std::env::var_os("CHERENKOV_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| util::root().join("target/release/cherenkov"))
}

pub fn model() -> Result<PathBuf> {
    PathBuf::from(
        std::env::var_os("CHERENKOV_MODEL_DIR")
            .context("set CHERENKOV_MODEL_DIR or use cargo xtask smoke --model PATH")?,
    )
    .canonicalize()
    .context("model directory")
}

pub fn wait_for(mut check: impl FnMut() -> Result<bool>, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if check()? {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    anyhow::bail!("server did not reach expected state within {timeout:?}")
}

pub struct Server {
    pub process: ChildGuard,
    pub directory: tempfile::TempDir,
    pub address: String,
    model: PathBuf,
    idle_seconds: u64,
}

impl Server {
    pub fn start(idle_seconds: u64) -> Result<Self> {
        let directory = tempfile::Builder::new()
            .prefix("cherenkov-smoke-")
            .tempdir_in("/private/tmp")?;
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))?;
        let model = model()?;
        let config = directory.path().join("server.toml");
        write_config(
            &config,
            &directory.path().join("control.sock"),
            &model,
            4,
            idle_seconds,
        )?;
        let log = fs::File::create(directory.path().join("server.log"))?;
        let process = ChildGuard(
            Command::new(binary())
                .args(["serve", "--config"])
                .arg(&config)
                .stdout(log.try_clone()?)
                .stderr(log)
                .spawn()?,
        );
        let mut server = Self {
            directory,
            process,
            address: String::new(),
            model,
            idle_seconds,
        };
        wait_for(
            || {
                ensure!(
                    server.process.0.try_wait()?.is_none(),
                    "server exited: {}",
                    fs::read_to_string(server.directory.path().join("server.log"))?
                );
                if !server.socket().exists() {
                    return Ok(false);
                }
                let stats = server.stats()?;
                if stats["ready"] != true {
                    return Ok(false);
                }
                server.address = stats["http_address"]
                    .as_str()
                    .context("HTTP address")?
                    .to_owned();
                Ok(true)
            },
            Duration::from_secs(180),
        )?;
        Ok(server)
    }

    pub fn socket(&self) -> PathBuf {
        self.directory.path().join("control.sock")
    }

    pub fn control(&self, args: &[&str]) -> Result<Value> {
        let output = Command::new(binary())
            .args(args)
            .arg("--socket")
            .arg(self.socket())
            .output()?;
        serde_json::from_str(&util::checked(output)?).context("control JSON")
    }

    pub fn stats(&self) -> Result<Value> {
        Ok(self.control(&["status", "--json"])?["stats"].clone())
    }

    pub fn rewrite(&self, tokens: u64) -> Result<()> {
        write_config(
            &self.directory.path().join("server.toml"),
            &self.socket(),
            &self.model,
            tokens,
            self.idle_seconds,
        )
    }
}

fn write_config(path: &Path, socket: &Path, model: &Path, tokens: u64, idle: u64) -> Result<()> {
    fs::write(
        path,
        format!(
            "[server]\nmodel_dir = {}\nsocket = {}\nport = 0\n[limits]\ncontext_tokens = 512\ncache_idle_seconds = {idle}\nmax_output_tokens = 8\n[defaults]\nmax_tokens = {tokens}\nno_eos = true\n",
            serde_json::to_string(model)?,
            serde_json::to_string(socket)?
        ),
    )?;
    Ok(())
}

pub struct Response {
    pub status: u16,
    pub headers: String,
    pub body: String,
}

impl Response {
    pub fn json(&self) -> Result<Value> {
        Ok(serde_json::from_str(&self.body)?)
    }
    pub fn success(&self) -> Result<Value> {
        ensure!(self.status == 200, "HTTP {}: {}", self.status, self.body);
        self.json()
    }
}

pub fn request(address: &str, path: &str, body: Option<&Value>) -> Result<Response> {
    let mut stream = TcpStream::connect(address)?;
    stream.set_read_timeout(Some(Duration::from_secs(300)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let data = body
        .map(serde_json::to_vec)
        .transpose()?
        .unwrap_or_default();
    write!(
        stream,
        "{} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        if body.is_some() { "POST" } else { "GET" },
        data.len()
    )?;
    stream.write_all(&data)?;
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes)?;
    let response = String::from_utf8(bytes)?;
    let (headers, body) = response.split_once("\r\n\r\n").context("HTTP headers")?;
    let status = headers
        .split_whitespace()
        .nth(1)
        .context("HTTP status")?
        .parse()?;
    Ok(Response {
        status,
        headers: headers.to_owned(),
        body: body.to_owned(),
    })
}

pub fn complete(address: &str, prompt: &str, extra: Value) -> Result<Response> {
    let mut body = json!({"model":"cherenkov","prompt":prompt});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().context("request extras")?.clone());
    request(address, "/v1/completions", Some(&body))
}

pub fn chat(address: &str, messages: &Value) -> Result<Value> {
    request(
        address,
        "/v1/chat/completions",
        Some(&json!({"model":"cherenkov","max_tokens":4,"messages":messages})),
    )?
    .success()
}

pub fn timed_command(mut command: Command, timeout: Duration) -> Result<String> {
    let out = tempfile::tempfile()?;
    let err = tempfile::tempfile()?;
    let mut child = ChildGuard(
        command
            .stdout(Stdio::from(out.try_clone()?))
            .stderr(Stdio::from(err.try_clone()?))
            .spawn()?,
    );
    let start = Instant::now();
    loop {
        if let Some(status) = child.0.try_wait()? {
            use std::io::{Seek, SeekFrom};
            let (mut out, mut err) = (out, err);
            out.seek(SeekFrom::Start(0))?;
            err.seek(SeekFrom::Start(0))?;
            let (mut stdout, mut stderr) = (String::new(), String::new());
            out.read_to_string(&mut stdout)?;
            err.read_to_string(&mut stderr)?;
            ensure!(status.success(), "command failed: {stderr}");
            return Ok(stdout.trim().to_owned());
        }
        ensure!(start.elapsed() < timeout, "command timed out");
        std::thread::sleep(Duration::from_millis(100));
    }
}
