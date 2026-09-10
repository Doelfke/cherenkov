//! Exercise real child processes without allocating GPU memory or reading weights.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Command, Output},
    time::{Duration, Instant},
};
use xtask::{capture, util};

struct Fixture {
    directory: tempfile::TempDir,
    binary: PathBuf,
    model: PathBuf,
    suite: PathBuf,
    output: PathBuf,
}

impl Fixture {
    fn new(cap: usize) -> Result<Self> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        let binary = root.join("engine");

        fs::write(
            &binary,
            "#!/bin/sh\ndir=$(dirname \"$0\")\necho $$ > \"$dir/pid\"\n[ -z \"${CHERENKOV_FN_FAKE+x}\" ] || exit 99\ncat \"$dir/answer\"\ncat \"$dir/telemetry\" >&2\n",
        )?;
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o700))?;
        fs::write(
            root.join("answer"),
            "A complete answer.\n<svg xmlns=\"http://www.w3.org/2000/svg\"><path/></svg>\n",
        )?;
        fs::write(
            root.join("telemetry"),
            include_str!("fixtures/telemetry.txt"),
        )?;

        let model = root.join("model");

        fs::create_dir_all(model.join("packed"))?;

        for name in ["config.json", "tokenizer.json", "packed/manifest.json"] {
            fs::write(model.join(name), "{}")?;
        }

        let suite = root.join("suite.json");

        util::write_json(
            &suite,
            &json!({
                "rounds": 1, "max_ctx": 512,
                "configurations": [{"id": "exact", "label": "4-bit", "args": [], "reproducible_cut": false}],
                "cases": [
                    {"id": "code", "kind": "decode", "prompt": "Explain.", "max_tokens": cap, "stop": "eos"},
                    {"id": "pelican", "kind": "svg", "prompt": "Draw.", "max_tokens": cap, "stop": "eos"}
                ]
            }),
        )?;

        let output = root.join("report");

        Ok(Self {
            directory,
            binary,
            model,
            suite,
            output,
        })
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));

        command
            .arg("bench")
            .arg(&self.model)
            .arg("--binary")
            .arg(&self.binary)
            .arg("--suite")
            .arg(&self.suite)
            .arg("--output")
            .arg(&self.output)
            .arg("--allow-battery")
            .env("CHERENKOV_FN_FAKE", "1");

        command
    }

    fn report(&self) -> Result<Value> {
        util::json(&self.output.join("report.json"))
    }
}

fn successful(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn suite_runs_resumes_and_keeps_svg_passive() -> Result<()> {
    let fixture = Fixture::new(128)?;

    successful(fixture.command().output()?);

    let first = fixture.report()?;

    assert_eq!(first["runs"].as_array().unwrap().len(), 2);
    assert!(
        first["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "ok")
    );
    assert_eq!(first["runs"][0]["metrics"]["prefill_seconds"], 5.0);
    assert!(fixture.output.join("pelicans/exact.svg").exists());
    successful(fixture.command().arg("--resume").output()?);
    assert_eq!(fixture.report()?["runs"], first["runs"]);

    let html = fs::read_to_string(fixture.output.join("gallery.html"))?;

    assert!(html.contains("<img ") && !html.contains("<svg"));

    Ok(())
}

#[test]
fn capped_answers_are_retained_but_excluded() -> Result<()> {
    let fixture = Fixture::new(64)?;

    successful(fixture.command().output()?);

    let report = fixture.report()?;

    assert!(
        report["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "incomplete")
    );
    assert!(xtask::report::rows(&report)?.is_empty());
    assert!(!fixture.output.join("pelicans/exact.svg").exists());

    Ok(())
}

#[test]
fn failed_memory_check_stops_then_resume_retries() -> Result<()> {
    let fixture = Fixture::new(128)?;
    let telemetry = include_str!("fixtures/telemetry.txt");

    fs::write(
        fixture.directory.path().join("telemetry"),
        telemetry.replace("20.98", "26.00"),
    )?;
    assert!(!fixture.command().output()?.status.success());

    let failed = fixture.report()?;

    assert_eq!(failed["runs"].as_array().unwrap().len(), 1);
    assert_eq!(failed["runs"][0]["status"], "failed");
    assert!(
        failed["runs"][0]["error"]
            .as_str()
            .unwrap()
            .contains("25 GB")
    );
    fs::write(fixture.directory.path().join("telemetry"), telemetry)?;
    successful(fixture.command().arg("--resume").output()?);
    assert!(
        fixture.report()?["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "ok")
    );

    Ok(())
}

#[test]
fn timeout_reaps_the_engine() -> Result<()> {
    let fixture = Fixture::new(128)?;
    let args = vec!["/bin/sleep".into(), "60".into()];
    let start = Instant::now();
    let result = capture::run(
        &args,
        &fixture.directory.path().join("out"),
        true,
        Some(Duration::from_millis(100)),
    )?;

    assert!(result.timed_out && result.code != 0);
    assert!(start.elapsed() < Duration::from_secs(5));

    Ok(())
}

#[test]
fn interrupt_reaps_the_engine_and_leaves_a_resumable_report() -> Result<()> {
    let fixture = Fixture::new(128)?;

    fs::write(
        &fixture.binary,
        "#!/bin/sh\ndir=$(dirname \"$0\")\necho $$ > \"$dir/pid\"\nexec /bin/sleep 60\n",
    )?;

    let mut parent = capture::ChildGuard(
        fixture
            .command()
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?,
    );
    let pid_path = fixture.directory.path().join("pid");
    let start = Instant::now();

    while !pid_path.exists() {
        assert!(start.elapsed() < Duration::from_secs(10));
        assert!(parent.0.try_wait()?.is_none());
        std::thread::sleep(Duration::from_millis(20));
    }

    let pid: libc::pid_t = fs::read_to_string(pid_path)?.trim().parse()?;

    // Signal only the runner: its child must be explicitly stopped during unwind.
    assert_eq!(
        unsafe { libc::kill(parent.0.id() as libc::pid_t, libc::SIGTERM) },
        0
    );

    let status = loop {
        if let Some(status) = parent.0.try_wait()? {
            break status;
        }

        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(20));
    };

    assert!(!status.success());
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
    assert_eq!(fixture.report()?["runs"], json!([]));
    assert!(fixture.output.join("outputs/r1-code-exact.txt").exists());

    Ok(())
}
