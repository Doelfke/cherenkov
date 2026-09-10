use crate::{
    bench::append,
    capture, metrics,
    report::{median, text},
    util,
};
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Args)]
pub struct Options {
    #[arg(long)]
    pub before: PathBuf,
    #[arg(long)]
    pub after: PathBuf,
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 2)]
    pub rounds: usize,
    #[arg(long,num_args=1..,default_values=["4","3","2"],value_parser=["4","3","2","4/3","4/2"])]
    pub configs: Vec<String>,
}

fn precision(row: &Value) -> (u64, u64) {
    let bits = row["bits"].as_u64().unwrap_or(4);
    (bits, row["miss_bits"].as_u64().unwrap_or(bits))
}

pub fn write_summary(out: &Path, report: &Value) -> Result<()> {
    let runs: Vec<_> = report["runs"]
        .as_array()
        .context("runs")?
        .iter()
        .filter(|r| r["valid"] == true)
        .collect();
    let mut configurations = Vec::new();
    for r in &runs {
        if !configurations.contains(&precision(r)) {
            configurations.push(precision(r));
        }
    }
    let mut s = format!(
        "# Low-bit prefill comparison\n\n{} valid samples. Medians exclude loading and conversion.\n\n| Prompt tokens | Resident / miss bits | Before pp/s | After pp/s | Change | Pairs |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n",
        runs.len()
    );
    let mut prompts: Vec<_> = report["prompts"]
        .as_object()
        .context("prompts")?
        .keys()
        .collect();
    prompts.sort_by_key(|p| p.parse::<usize>().unwrap_or(usize::MAX));
    for prompt in prompts {
        for &(bits, miss) in &configurations {
            let selected: Vec<_> = runs
                .iter()
                .filter(|r| r["prompt"] == *prompt && precision(r) == (bits, miss))
                .copied()
                .collect();
            append_median(&mut s, &selected, bits, miss)?;
        }
    }
    if let Some(notes) = report["notes"].as_array() {
        s.push('\n');
        for note in notes {
            writeln!(s, "{}", text(note))?;
        }
    }
    let controls: Vec<_> = runs
        .iter()
        .filter(|r| precision(r) == (4, 4) && r["binary"] == "before")
        .collect();
    if !controls.is_empty() {
        s.push_str("\n## Output checks\n\n");
    }
    for before in controls {
        let after = runs.iter().find(|r| {
            r["round"] == before["round"]
                && r["prompt"] == before["prompt"]
                && precision(r) == (4, 4)
                && r["binary"] == "after"
        });
        let Some(after) = after else {
            continue;
        };
        let same = fs::read(out.join(text(&before["output"])))?
            == fs::read(out.join(text(&after["output"])))?;
        writeln!(
            s,
            "- Round {}, {} tokens: Q4 output {}.",
            before["round"],
            before["metrics"]["prompt_tokens"],
            if same { "identical" } else { "DIFFERS" }
        )?;
    }
    s.push_str(
        "\nBinary hashes, timings, source and power readings are in [report.json](report.json).\n",
    );
    fs::write(out.join("README.md"), s)?;
    Ok(())
}

fn append_median(s: &mut String, runs: &[&Value], bits: u64, miss: u64) -> Result<()> {
    let rates = |label: &str| {
        runs.iter()
            .filter(|r| r["binary"] == label)
            .filter_map(|r| r["metrics"]["pp_s"].as_f64())
            .collect::<Vec<_>>()
    };
    let (a, b) = (rates("before"), rates("after"));
    if a.is_empty() || b.is_empty() {
        return Ok(());
    }
    let pairs = a.len().min(b.len());
    let (a, b) = (median(a), median(b));
    let label = if bits == miss {
        bits.to_string()
    } else {
        format!("{bits} / {miss}")
    };
    writeln!(
        s,
        "| {} | {label} | {a:.1} | {b:.1} | {:+.1}% | {pairs} |",
        runs[0]["metrics"]["prompt_tokens"],
        100.0 * (b / a - 1.0)
    )?;
    Ok(())
}

struct Sample<'a> {
    options: &'a Options,
    binary: &'a Path,
    label: &'a str,
    round: usize,
    length: usize,
    config: &'a str,
    prompt: &'a str,
}

impl Sample<'_> {
    fn run(&self) -> Result<Value> {
        let (bits, miss) = self
            .config
            .split_once('/')
            .unwrap_or((self.config, self.config));
        let name = format!(
            "r{}-repeat{}-q{}-{}",
            self.round + 1,
            self.length,
            self.config.replace('/', "-miss"),
            self.label
        );
        let args = vec![
            self.binary.display().to_string(),
            self.options.model.canonicalize()?.display().to_string(),
            self.prompt.into(),
            "--experts".into(),
            bits.into(),
            "--miss-experts".into(),
            miss.into(),
            "--max-ctx".into(),
            "8192".into(),
            "--max-tokens".into(),
            "8".into(),
            "--no-eos".into(),
        ];
        let c = capture::run(
            &args,
            &self.options.out.join(format!("{name}.txt")),
            false,
            Some(Duration::from_secs(300)),
        )?;
        let mut row = json!({
            "id": name,
            "round": self.round + 1,
            "binary": self.label,
            "bits": bits.parse::<u8>()?,
            "miss_bits": miss.parse::<u8>()?,
            "prompt": self.length.to_string(),
            "args": args,
            "wall_seconds": c.wall_seconds,
            "power_before": c.power_before,
            "power_after": c.power_after,
            "power_samples": c.power_samples,
            "exit_code": c.code,
            "stderr": c.stderr,
            "output": format!("{name}.txt"),
            "valid": false,
        });
        if let Err(error) = validate(&c, &mut row) {
            row["error"] = json!(error.to_string());
        }
        eprintln!(
            "{name}: {} pp/s, valid={}",
            row["metrics"]["pp_s"], row["valid"]
        );
        Ok(row)
    }
}

fn validate(c: &capture::Captured, row: &mut Value) -> Result<()> {
    ensure!(
        c.code == 0 && !c.timed_out && c.cycle.is_none(),
        "engine failed, timed out, or cycled"
    );
    let m = metrics::parse(&c.stderr)?;
    row["metrics"] = m.clone();
    ensure!(m["store_build_seconds"].is_null(), "sample built a store");
    ensure!(
        c.stable_power() && c.power_after["source"] == "ac",
        "power changed"
    );
    ensure!(
        m["metal_gb"].as_f64().unwrap() <= 25.0,
        "reported Metal allocations exceeded 25 GB"
    );
    ensure!(
        m["output_tokens"] == 8,
        "decode handoff did not emit eight tokens"
    );
    row["valid"] = json!(true);
    Ok(())
}

pub fn run(options: Options) -> Result<()> {
    ensure!(options.rounds > 0, "rounds must be positive");
    ensure!(
        options.configs.iter().collect::<HashSet<_>>().len() == options.configs.len(),
        "duplicate configurations"
    );
    fs::create_dir_all(&options.out)?;
    ensure!(
        !options.out.join("report.json").exists(),
        "output already contains a report"
    );
    let before = options.before.canonicalize()?;
    let after = options.after.canonicalize()?;
    let mut prompts = json!({});
    let sentence = "The quick brown fox jumps over the lazy dog. Each sentence is part of a repeated document used to measure prompt processing.\n";
    for n in [4, 20, 128] {
        prompts[n.to_string()] =
            json!(sentence.repeat(n) + "\nSummarize this document in three sentences.");
    }
    let mut report = json!({
        "binaries": {
            "before": {"path": before, "sha256": util::digest(&before)?},
            "after": {"path": after, "sha256": util::digest(&after)?},
        },
        "source_commit": util::output(&["git", "rev-parse", "HEAD"])?,
        "source_diff": util::output(&["git", "diff", "HEAD"])?,
        "prompts": prompts,
        "configurations": options.configs,
        "model": options.model.canonicalize()?,
        "notes": [
            "Fresh processes, context 8192, adaptive pool and two adaptive drafts.",
            "Eight decode tokens check handoff; no prefix cache or deadline cut.",
            "Loading and conversion excluded; any store build invalidates a sample.",
            "AC checked at process boundaries and every 30 seconds.",
            "Configurations and binary order rotate; OS file caches are retained.",
        ],
        "runs": [],
    });
    for round in 0..options.rounds {
        for length in [4, 20, 128] {
            for index in 0..options.configs.len() {
                let config = &options.configs[(index + round) % options.configs.len()];
                let order = if round % 2 == 0 {
                    [("before", &before), ("after", &after)]
                } else {
                    [("after", &after), ("before", &before)]
                };
                for (label, binary) in order {
                    let sample = Sample {
                        options: &options,
                        binary,
                        label,
                        round,
                        length,
                        config,
                        prompt: prompts[&length.to_string()].as_str().unwrap(),
                    };
                    let row = sample.run()?;
                    let valid = row["valid"] == true;
                    let error = row["error"].clone();
                    append(&mut report, "runs", row);
                    util::write_json(&options.out.join("report.json"), &report)?;
                    write_summary(&options.out, &report)?;
                    ensure!(valid, "{error}; details saved in {}", options.out.display());
                }
            }
        }
    }
    Ok(())
}
