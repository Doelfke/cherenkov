use anyhow::{Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use std::{collections::HashSet, io::Write, path::PathBuf, process::Stdio};
use xtask::{bench, prefill, report, util};

// Compile exactly the same fragments as the engine; no source-text parser.
#[path = "../../src/kernels.rs"]
mod kernels;

#[derive(Parser)]
#[command(about = "Cherenkov repository tasks")]
struct Cli {
    #[command(subcommand)]
    command: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Run the complete-answer benchmark suite and pelican gallery.
    Bench(bench::Options),
    /// Compare prefill throughput between two executables.
    Prefill(prefill::Options),
    /// Regenerate a saved report without loading a model.
    Summarize { directory: PathBuf },
    /// Rebuild the README benchmark section from a completed saved run.
    Readme { directory: PathBuf },
    /// Syntax-check all assembled Metal libraries.
    CheckMetal,
    /// Run explicit live smoke tests (starts its own server).
    Smoke {
        #[arg(value_enum,default_value_t=Smoke::All)]
        kind: Smoke,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        binary: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Smoke {
    All,
    Server,
    Control,
    Sessions,
    Download,
}

fn check_metal() -> Result<()> {
    let mut failed = false;

    for (name, source) in [
        ("FORWARD_MSL", kernels::FORWARD_MSL),
        ("BATCH_MSL", kernels::BATCH_MSL),
        ("CLOCK_PROBE_MSL", kernels::CLOCK_PROBE_MSL),
    ] {
        let mut input = tempfile::NamedTempFile::new()?;

        input.write_all(source.as_bytes())?;

        let child = util::command(&[
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-fsyntax-only",
            "-x",
            "metal",
        ])
        .arg(input.path())
        .current_dir(util::root().join("kernels"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
        let output = child.wait_with_output()?;

        println!(
            "{} {name}",
            if output.status.success() {
                "ok"
            } else {
                "FAIL"
            }
        );

        failed |= !output.status.success();

        print_diagnostics(&String::from_utf8_lossy(&output.stderr));
    }

    ensure!(!failed, "Metal syntax check failed");

    Ok(())
}

fn print_diagnostics(stderr: &str) {
    let mut seen = HashSet::new();
    let mut keep = true;

    for line in stderr.lines() {
        if line.contains(": error:") || line.contains(": warning:") {
            keep = seen.insert(line);
        }

        if keep {
            eprintln!("{line}");
        }
    }
}

fn smoke(kind: Smoke, model: Option<PathBuf>, binary: Option<PathBuf>) -> Result<()> {
    if binary.is_none() {
        util::build()?;
    }

    let mut command = util::command(&["cargo", "test", "-p", "xtask", "--test", "smoke"]);
    let name = match kind {
        Smoke::All => None,
        Smoke::Server => Some("server_smoke"),
        Smoke::Control => Some("control_smoke"),
        Smoke::Sessions => Some("concurrency::"),
        Smoke::Download => Some("download_smoke"),
    };

    if let Some(name) = name {
        command.arg(name);
    }

    command.args(["--", "--ignored", "--test-threads=1", "--nocapture"]);

    if let Some(model) = model {
        command.env("CHERENKOV_MODEL_DIR", model.canonicalize()?);
    }

    if let Some(binary) = binary {
        command.env("CHERENKOV_BINARY", binary.canonicalize()?);
    }

    ensure!(command.status()?.success(), "smoke tests failed");

    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Task::Bench(options) => {
            xtask::capture::install_interrupt_handler()?;

            bench::run(options)
        }
        Task::Prefill(options) => {
            xtask::capture::install_interrupt_handler()?;

            prefill::run(options)
        }
        Task::Summarize { directory } => {
            let data = util::json(&directory.join("report.json"))?;

            if data.get("prompts").is_some() {
                prefill::write_summary(&directory, &data)
            } else {
                report::write(&directory, &data)
            }
        }
        Task::Readme { directory } => xtask::readme::update(&directory),
        Task::CheckMetal => check_metal(),
        Task::Smoke {
            kind,
            model,
            binary,
        } => smoke(kind, model, binary),
    }
}
