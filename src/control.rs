//! Local control: one bounded JSON exchange per connection, authenticated by UID.

use crate::units::BYTES_PER_KIB;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub mod state;
pub use state::State;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Status,
    ConfigShow,
    ConfigReload,
}

const MAX_FRAME: usize = 64 * BYTES_PER_KIB;
const TIMEOUT: Duration = Duration::from_secs(2);

/// The deadline applies to the whole frame, even if a client dribbles bytes.
fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>> {
    let deadline = Instant::now() + TIMEOUT;
    let mut bytes = Vec::new();

    loop {
        wait_readable(stream, deadline)?;

        let mut chunk = [0; BYTES_PER_KIB];
        let n = stream.read(&mut chunk).context("reading control frame")?;

        ensure!(n > 0, "incomplete control frame");

        let end = chunk[..n].iter().position(|&b| b == b'\n');

        bytes.extend_from_slice(&chunk[..end.unwrap_or(n)]);
        ensure!(bytes.len() <= MAX_FRAME, "control frame exceeds 64 KiB");

        if end.is_some() {
            return Ok(bytes);
        }
    }
}

fn wait_readable(stream: &UnixStream, deadline: Instant) -> Result<()> {
    // macOS rejects timeout changes after peer close, even with unread data.
    // Poll also bounds the entire frame. This connection has only one reader.
    let mut fd = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };

    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .context("control request timed out")?;
        // TIMEOUT is two seconds; round up to poll's millisecond precision.
        let ready = unsafe { libc::poll(&mut fd, 1, remaining.as_millis() as i32 + 1) };

        if ready > 0 {
            return Ok(());
        }

        ensure!(ready != 0, "control request timed out");

        let error = std::io::Error::last_os_error();

        if error.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }

        return Err(error).context("polling control socket");
    }
}

fn write_frame(stream: &mut UnixStream, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;

    ensure!(bytes.len() <= MAX_FRAME, "control response exceeds 64 KiB");
    bytes.push(b'\n');
    stream
        .set_write_timeout(Some(TIMEOUT))
        .context("setting control write timeout")?;
    stream.write_all(&bytes).context("writing control frame")?;

    Ok(())
}

fn same_user(stream: &UnixStream) -> Result<()> {
    let (mut uid, mut gid) = (0, 0);
    let result = unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) };

    ensure!(result == 0, "cannot determine control peer identity");
    ensure!(
        uid == unsafe { libc::geteuid() },
        "control peer must have the server's UID"
    );

    Ok(())
}

pub fn query(socket: &Path, command: Command) -> Result<Value> {
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("connecting to {}", socket.display()))?;

    same_user(&stream)?;
    write_frame(&mut stream, &serde_json::to_value(command)?)?;

    let response: Value = serde_json::from_slice(&read_frame(&mut stream)?)?;

    ensure!(
        response["ok"] == true,
        "{}",
        response["error"]
            .as_str()
            .unwrap_or("invalid control response")
    );

    Ok(response["data"].clone())
}

fn handle(mut stream: UnixStream, state: &State) -> Result<()> {
    same_user(&stream)?;

    let result = (|| -> Result<Value> {
        Ok(
            match serde_json::from_slice::<Command>(&read_frame(&mut stream)?)? {
                Command::Status => state.status(),
                Command::ConfigShow => state.show_config(),
                Command::ConfigReload => state.reload()?,
            },
        )
    })();
    let (data, error) = match result {
        Ok(data) => (Some(data), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };

    write_frame(
        &mut stream,
        &json!({"ok": error.is_none(), "data": data, "error": error}),
    )
}

pub struct Listener {
    socket: PathBuf,
    identity: (u64, u64),
    // Held until the socket is removed; serializes startup and stale cleanup.
    _lock: File,
    stopped: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Listener {
    pub fn start(socket: &Path, state: Arc<State>) -> Result<Self> {
        let parent = socket.parent().context("socket needs a parent directory")?;

        match std::fs::DirBuilder::new().mode(0o700).create(parent) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e).context("creating control directory"),
        }

        let uid = unsafe { libc::geteuid() };
        let dir = std::fs::symlink_metadata(parent)?;

        ensure!(
            dir.is_dir() && dir.uid() == uid && dir.mode() & 0o777 == 0o700,
            "control directory must be owned by this user, not a symlink, and mode 0700: {}",
            parent.display()
        );

        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(socket.with_extension("lock"))?;
        let meta = lock.metadata()?;

        ensure!(
            meta.is_file() && meta.uid() == uid && meta.mode() & 0o777 == 0o600,
            "unsafe control lock file"
        );
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another server owns control socket {}",
            socket.display()
        );

        match std::fs::symlink_metadata(socket) {
            Ok(meta) => {
                ensure!(
                    meta.file_type().is_socket() && meta.uid() == uid,
                    "refusing to replace non-socket control path"
                );

                match UnixStream::connect(socket) {
                    Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                        std::fs::remove_file(socket)?
                    }
                    _ => anyhow::bail!("control socket already in use: {}", socket.display()),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }

        let listener = UnixListener::bind(socket).context("binding control socket")?;

        std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;

        let meta = std::fs::symlink_metadata(socket)?;
        let stopped = Arc::new(AtomicBool::new(false));
        let stop = stopped.clone();
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stop.load(Ordering::Acquire) {
                    break;
                }

                if let Ok(stream) = stream {
                    let _ = handle(stream, &state);
                }
            }
        });

        Ok(Self {
            socket: socket.to_owned(),
            identity: (meta.dev(), meta.ino()),
            _lock: lock,
            stopped,
            thread: Some(thread),
        })
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);

        let _ = UnixStream::connect(&self.socket);

        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }

        if let Ok(meta) = std::fs::symlink_metadata(&self.socket)
            && (meta.dev(), meta.ino()) == self.identity
        {
            let _ = std::fs::remove_file(&self.socket);
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/control.rs"]
mod tests;
