use super::*;
use crate::config::{Config, Overrides, Source};
use std::sync::atomic::AtomicUsize;

#[test]
fn buffered_control_response_survives_peer_close() {
    let (mut client, mut server) = UnixStream::pair().unwrap();
    server.write_all(b"{\"ok\":true}\n").unwrap();
    drop(server);
    assert_eq!(read_frame(&mut client).unwrap(), b"{\"ok\":true}");
}

#[test]
fn idle_control_reads_respect_the_frame_deadline() {
    let (client, _peer) = UnixStream::pair().unwrap();
    let error = wait_readable(&client, Instant::now() + Duration::from_millis(10)).unwrap_err();
    assert!(error.to_string().contains("timed out"));
}
static SERIAL: AtomicUsize = AtomicUsize::new(0);

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        // Keep Unix socket paths below macOS's sockaddr_un path limit.
        let path = PathBuf::from(format!(
            "/private/tmp/cherenkov-test-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .unwrap();
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn state() -> Arc<State> {
    Arc::new(State::new(
        Source {
            path: None,
            overrides: Overrides::default(),
        },
        Config::default(),
    ))
}

#[test]
fn socket_queries_work_and_reject_mutations_without_a_config_file() {
    let dir = Directory::new();
    let socket = dir.0.join("control.sock");
    let state = state();
    state.update(|s| {
        s.ready = true;
        s.completed_requests = 3;
    });
    let listener = Listener::start(&socket, state).unwrap();
    assert_eq!(std::fs::metadata(&socket).unwrap().mode() & 0o777, 0o600);
    assert_eq!(
        query(&socket, Command::Status).unwrap()["stats"]["completed_requests"],
        3
    );
    assert_eq!(
        query(&socket, Command::ConfigShow).unwrap()["effective"]["generation"],
        1
    );
    assert!(query(&socket, Command::ConfigReload).is_err());
    assert!(Listener::start(&socket, self::state()).is_err());
    drop(listener);
    assert!(!socket.exists());
}

#[test]
fn reload_is_atomic_preserves_overrides_and_keeps_old_snapshots() {
    let dir = Directory::new();
    let path = dir.0.join("config.toml");
    std::fs::write(&path, "[defaults]\nmax_tokens = 128").unwrap();
    let source = Source {
        path: Some(path.clone()),
        overrides: Overrides {
            max_tokens: Some(64),
            ..Overrides::default()
        },
    };
    let state = State::new(source.clone(), source.resolve().unwrap());
    let old = state.config();
    std::fs::write(&path, "[defaults]\nmax_tokens = 256\nstream = true").unwrap();
    state.reload().unwrap();
    assert_eq!(state.config().config.defaults.max_tokens, 64);
    assert!(state.config().config.defaults.stream);
    assert!(!old.config.defaults.stream);
    assert_eq!(state.config().generation, 2);
    for text in [
        "[defaults]\nstream = false\n[experts]\nmiss_bits = 2",
        "[defaults]\ntemperature = 0.7",
        "invalid toml",
    ] {
        std::fs::write(&path, text).unwrap();
        assert!(state.reload().is_err());
        assert!(state.config().config.defaults.stream);
        assert_eq!(state.config().generation, 2);
    }
}

#[test]
fn socket_startup_refuses_unsafe_paths_and_recovers_a_stale_socket() {
    let dir = Directory::new();
    let socket = dir.0.join("control.sock");
    std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(Listener::start(&socket, state()).is_err());
    std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(&socket, "keep").unwrap();
    assert!(Listener::start(&socket, state()).is_err());
    assert_eq!(std::fs::read_to_string(&socket).unwrap(), "keep");
    std::fs::remove_file(&socket).unwrap();
    drop(UnixListener::bind(&socket).unwrap());
    let listener = Listener::start(&socket, state()).unwrap();
    assert!(query(&socket, Command::Status).is_ok());
    drop(listener);
    let link = dir.0.join("link");
    std::os::unix::fs::symlink(&dir.0, &link).unwrap();
    assert!(Listener::start(&link.join("control.sock"), state()).is_err());
}

#[test]
fn malformed_and_oversized_control_messages_are_bounded() {
    let dir = Directory::new();
    let socket = dir.0.join("control.sock");
    let _listener = Listener::start(&socket, state()).unwrap();
    for input in [
        "{\"op\":\"unknown\"}\n".to_owned(),
        "x".repeat(MAX_FRAME + 1) + "\n",
    ] {
        let mut stream = UnixStream::connect(&socket).unwrap();
        stream.write_all(input.as_bytes()).unwrap();
        let response: Value = serde_json::from_slice(&read_frame(&mut stream).unwrap()).unwrap();
        assert_eq!(response["ok"], false);
    }
    assert!(query(&socket, Command::Status).is_ok());
}
