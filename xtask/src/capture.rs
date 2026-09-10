//! File-backed child output keeps long generations out of memory while running.
use crate::{metrics, util};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
    process::{Child, Command},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

/// Let the polling loop unwind normally so its child is killed and reaped.
pub fn install_interrupt_handler() -> Result<()> {
    for signal in [libc::SIGINT, libc::SIGTERM] {
        // The handler only writes a lock-free atomic; it allocates nothing.
        let previous =
            unsafe { libc::signal(signal, interrupt as *const () as libc::sighandler_t) };
        ensure!(
            previous != libc::SIG_ERR,
            "could not install interrupt handler"
        );
    }
    Ok(())
}

pub fn power() -> Value {
    let detail = util::output(&["pmset", "-g", "batt"]).unwrap_or_else(|_| "unknown".into());
    let source = if detail.contains("'AC Power'") {
        "ac"
    } else if detail.contains("'Battery Power'") {
        "battery"
    } else {
        "unknown"
    };
    json!({"source":source,"detail":detail})
}

pub struct ChildGuard(pub Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub struct Captured {
    pub code: i32,
    pub stderr: String,
    pub wall_seconds: f64,
    pub power_before: Value,
    pub power_after: Value,
    pub power_samples: Vec<Value>,
    pub cycle: Option<Value>,
    pub timed_out: bool,
}

impl Captured {
    pub fn stable_power(&self) -> bool {
        self.power_samples
            .iter()
            .chain([&self.power_after])
            .all(|p| p["source"] == self.power_before["source"])
    }
}

pub fn run(
    args: &[String],
    output: &Path,
    allow_battery: bool,
    timeout: Option<Duration>,
) -> Result<Captured> {
    let before = power();
    ensure!(
        allow_battery || before["source"] == "ac",
        "AC power is required; reconnect and resume, or use --allow-battery"
    );
    let mut diagnostics = tempfile::tempfile()?;
    let mut command = Command::new(&args[0]);
    command.args(&args[1..]).current_dir(util::root());
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("CHERENKOV_") {
            command.env_remove(name);
        }
    }
    let started = Instant::now();
    let mut child = ChildGuard(
        command
            .stdout(File::create(output)?)
            .stderr(diagnostics.try_clone()?)
            .spawn()?,
    );
    let mut samples = vec![before.clone()];
    let mut last_poll = Instant::now();
    let mut cycle = None;
    let mut timed_out = false;
    let code = loop {
        ensure!(
            !INTERRUPTED.load(Ordering::Relaxed),
            "interrupted; current output retained, resume to retry"
        );
        if let Some(status) = child.0.try_wait()? {
            break status.code().unwrap_or(-1);
        }
        if timeout.is_some_and(|t| started.elapsed() >= t) {
            timed_out = true;
            child.0.kill()?;
            break child.0.wait()?.code().unwrap_or(-1);
        }
        if last_poll.elapsed() >= Duration::from_secs(30) {
            samples.push(power());
            let bytes = fs::read(output)?;
            cycle = metrics::repeated_tail(&String::from_utf8_lossy(&bytes));
            if cycle.is_some() {
                child.0.kill()?;
                break child.0.wait()?.code().unwrap_or(-1);
            }
            eprintln!(
                "{:.0}s elapsed: {}",
                started.elapsed().as_secs_f64(),
                output.display()
            );
            last_poll = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    diagnostics.seek(SeekFrom::Start(0))?;
    let mut stderr = String::new();
    diagnostics.read_to_string(&mut stderr)?;
    Ok(Captured {
        code,
        stderr,
        wall_seconds: started.elapsed().as_secs_f64(),
        power_before: before,
        power_after: power(),
        power_samples: samples,
        cycle,
        timed_out,
    })
}
