use crate::{
    capture, metrics, report,
    suite::{self, Case, Configuration, Suite},
    util,
};
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Args)]
pub struct Options {
    pub model_dir: PathBuf,
    #[arg(long, default_value = "benchmarks/suite.json")]
    pub suite: PathBuf,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub configs: Option<String>,
    #[arg(long)]
    pub cases: Option<String>,
    #[arg(long)]
    pub rounds: Option<usize>,
    #[arg(long)]
    pub case_cap: Vec<String>,
    #[arg(long)]
    pub binary: Option<PathBuf>,
    #[arg(long)]
    pub resume: bool,
    #[arg(long)]
    pub build_stores: bool,
    #[arg(long)]
    pub allow_battery: bool,
    #[arg(long)]
    pub dry_run: bool,
    /// Rebuild the README benchmark section after the suite finishes.
    #[arg(long)]
    pub update_readme: bool,
}

fn set_caps(cases: &mut [Case], overrides: &[String]) -> Result<()> {
    for value in overrides {
        let (id, limit) = value
            .split_once('=')
            .context("--case-cap requires ID=TOKENS")?;
        let limit = limit.parse()?;

        ensure!(limit > 0, "token cap must be positive");

        cases
            .iter_mut()
            .find(|c| c.id == id)
            .with_context(|| format!("unknown case: {id}"))?
            .max_tokens = limit;
    }

    Ok(())
}

fn check_stores(model: &Path, configs: &[Configuration], allow_build: bool) -> Result<()> {
    ensure!(
        model.join("packed/manifest.json").exists(),
        "model must already be packed"
    );

    for config in configs {
        let Some(bits) = config.store_bits else {
            continue;
        };
        let present = model.join(format!("packed/experts{bits}.bin")).exists()
            && model.join(format!("packed/manifest{bits}.json")).exists();

        ensure!(
            present || allow_build,
            "{bits}-bit store missing; select cached configurations or pass --build-stores"
        );

        if !present {
            eprintln!("Allowing first-use {bits}-bit construction, reported in load time.");
        }
    }

    Ok(())
}

pub fn raise_saved_caps(out: &Path, report: &mut Value, signature: Value) -> Result<()> {
    let revision = json!({"previous_signature":report["signature"],"utc_changed":util::utc()?,"reason":"Raised answer safety caps; prompts, context, binary, and model unchanged."});

    append(report, "suite_revisions", revision);

    let runs = std::mem::take(report["runs"].as_array_mut().context("report runs")?);

    for mut record in runs {
        let args = record["args"].as_array().context("sample args")?;
        let pos = args
            .iter()
            .position(|a| a == "--max-tokens")
            .context("sample token cap")?;
        let old_cap = args[pos + 1]
            .as_str()
            .context("sample cap value")?
            .parse::<u64>()?;
        let cap = signature["cases"]
            .as_array()
            .context("signature cases")?
            .iter()
            .find(|c| c["id"] == record["case"])
            .context("case missing")?["max_tokens"]
            .as_u64()
            .context("token cap")?;

        if record["status"] != "incomplete" || cap <= old_cap {
            append(report, "runs", record);

            continue;
        }

        let archive = format!(
            "attempts/{}-cap-{old_cap}.txt",
            record["id"].as_str().context("sample id")?
        );

        fs::create_dir_all(out.join("attempts"))?;
        fs::rename(
            out.join(record["output"].as_str().context("sample output")?),
            out.join(&archive),
        )?;

        record["output"] = json!(archive);

        append(report, "previous_attempts", record);
    }

    report["cases"] = signature["cases"].clone();
    report["signature"] = signature;

    Ok(())
}

pub fn append(report: &mut Value, key: &str, value: Value) {
    report
        .as_object_mut()
        .unwrap()
        .entry(key.to_owned())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .unwrap()
        .push(value);
}

fn record_result(
    out: &Path,
    config: &Configuration,
    case: &Case,
    capture: &capture::Captured,
    record: &mut Value,
    allow_build: bool,
) -> Result<()> {
    if let Some(cycle) = &capture.cycle {
        record["status"] = json!("cycling");

        anyhow::bail!(
            "stopped after four repeated word blocks: {}",
            cycle["example"]
        );
    }

    ensure!(
        capture.code == 0,
        "engine exited {}: {}",
        capture.code,
        capture
            .stderr
            .chars()
            .rev()
            .take(3000)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );

    let m = metrics::parse(&capture.stderr)?;
    record["metrics"] = m.clone();

    ensure!(
        m["metal_gb"].as_f64().unwrap() <= 25.0,
        "reported Metal allocations exceeded 25 GB"
    );
    ensure!(
        capture.stable_power(),
        "power source changed during this sample"
    );
    ensure!(
        allow_build || m["store_build_seconds"].is_null(),
        "unexpected store construction; use --build-stores"
    );

    let reason = if m["output_tokens"].as_u64().unwrap() >= case.max_tokens as u64 {
        "length"
    } else {
        "eos"
    };
    record["finish_reason"] = json!(reason);

    if case.stop == "eos" && reason != "eos" {
        record["status"] = json!("incomplete");

        anyhow::bail!("incomplete answer: reached the token safety cap before EOS");
    }

    ensure!(
        case.stop != "length" || reason == "length",
        "fixed-length decode did not reach its token count"
    );

    if case.kind == "svg" {
        record["status"] = json!("invalid_svg");
        let text = fs::read_to_string(out.join(record["output"].as_str().unwrap()))?;
        let (svg, elements) = metrics::extract_svg(&text)?;
        let path = format!("pelicans/{}.svg", config.id);

        fs::write(out.join(&path), format!("{svg}\n"))?;

        record["svg"] = json!(path);
        record["svg_elements"] = json!(elements);
        record["within_element_budget"] = json!(elements <= 120);
    }

    record["status"] = json!("ok");

    Ok(())
}

struct Run<'a> {
    binary: &'a Path,
    model: &'a Path,
    out: &'a Path,
    options: &'a Options,
    max_ctx: usize,
}

impl Run<'_> {
    fn sample(&self, config: &Configuration, case: &Case, round: usize) -> Result<Value> {
        let id = format!("r{}-{}-{}", round + 1, case.id, config.id);
        let output = format!("outputs/{id}.txt");
        let mut args = vec![
            self.binary.display().to_string(),
            self.model.display().to_string(),
            case.prompt(),
        ];

        args.extend(config.args.clone());
        args.extend([
            "--max-tokens".into(),
            case.max_tokens.to_string(),
            "--max-ctx".into(),
            case.max_ctx.unwrap_or(self.max_ctx).to_string(),
        ]);

        if case.stop == "length" {
            args.push("--no-eos".into());
        }

        eprintln!("[{id}] starting");

        let present = config
            .store_bits
            .is_none_or(|b| self.model.join(format!("packed/experts{b}.bin")).exists());
        let c = capture::run(
            &args,
            &self.out.join(&output),
            self.options.allow_battery,
            None,
        )?;
        let mut record = json!({
            "id": id,
            "configuration": config.id,
            "case": case.id,
            "round": round + 1,
            "args": args,
            "output": output,
            "power_before": c.power_before,
            "power_after": c.power_after,
            "power_samples": c.power_samples,
            "status": "failed",
            "store_present_before": present,
            "wall_seconds": c.wall_seconds,
            "exit_code": c.code,
        });

        if let Some(cycle) = &c.cycle {
            record["cycle"] = cycle.clone();
        }

        if let Err(error) = record_result(
            self.out,
            config,
            case,
            &c,
            &mut record,
            self.options.build_stores,
        ) {
            record["error"] = json!(error.to_string());
        }

        eprintln!(
            "[{id}] {}, tg/s {}",
            record["status"], record["metrics"]["tg_s"]
        );

        Ok(record)
    }
}

fn new_report(
    signature: Value,
    configs: &[Configuration],
    cases: &[Case],
    binary: &Path,
    model: &Path,
) -> Result<Value> {
    let status = util::output(&["git", "status", "--porcelain"])?;

    Ok(json!({
        "version": 1,
        "signature": signature,
        "configurations": configs,
        "cases": cases,
        "runs": [],
        "provenance": {
            "commit": util::output(&["git", "rev-parse", "HEAD"])?,
            "dirty": !status.is_empty(),
            "git_status": status,
            "source_sha256": util::source_digest()?,
            "binary_sha256": util::digest(binary)?,
            "binary": binary,
            "model": model,
            "model_metadata_sha256": signature["model_metadata_sha256"],
            "utc_started": util::utc()?,
            "platform": util::output(&["uname", "-a"])?,
            "hardware": util::output(&["sysctl", "-n", "machdep.cpu.brand_string"])?,
            "memory_bytes": util::output(&["sysctl", "-n", "hw.memsize"])?,
            "rustc": util::output(&["rustc", "--version"])?,
            "power": capture::power(),
            "method": "fresh processes; rotated interleaved rounds; separate load/prefill/decode; no check; no prefix cache; complete answers through EOS; SVG phase after timings"
        }
    }))
}

pub fn run(options: Options) -> Result<()> {
    let suite: Suite = serde_json::from_value(util::json(&options.suite)?)?;
    let configs = suite::select(&suite.configurations, options.configs.as_deref(), |c| &c.id)?;
    let mut cases = suite::select(&suite.cases, options.cases.as_deref(), |c| &c.id)?;

    set_caps(&mut cases, &options.case_cap)?;

    let rounds = options.rounds.unwrap_or(suite.rounds);

    ensure!(
        rounds > 0 && !configs.is_empty() && !cases.is_empty(),
        "rounds and selections must be nonempty"
    );

    let jobs = suite::schedule(&configs, &cases, rounds);

    if options.dry_run {
        let plan: Vec<_> = jobs
            .iter()
            .map(|&(round, config, case)| {
                json!({
                    "round": round + 1,
                    "config": configs[config].id,
                    "case": cases[case].id,
                    "max_tokens": cases[case].max_tokens,
                })
            })
            .collect();

        println!("{}", serde_json::to_string_pretty(&plan)?);

        return Ok(());
    }

    let model = options.model_dir.canonicalize()?;

    check_stores(&model, &configs, options.build_stores)?;

    if options.binary.is_none() {
        util::build()?;
    }

    let binary = options
        .binary
        .clone()
        .unwrap_or_else(|| util::root().join("target/release/cherenkov"))
        .canonicalize()?;
    let out = util::absolute(
        &options
            .output
            .clone()
            .unwrap_or(util::root().join("results").join(util::utc()?)),
    )?;
    let mut metadata = json!({});

    for name in ["config.json", "tokenizer.json", "packed/manifest.json"] {
        metadata[name] = json!(util::digest(&model.join(name))?);
    }

    let signature = json!({
        "model_metadata_sha256": metadata,
        "binary_sha256": util::digest(&binary)?,
        "suite_sha256": util::digest(&options.suite)?,
        "model": model,
        "configs": configs,
        "cases": cases,
        "rounds": rounds,
        "allow_battery": options.allow_battery,
    });
    let mut report = prepare_report(&out, &options, signature, &configs, &cases, &binary, &model)?;

    fs::create_dir_all(out.join("outputs"))?;
    fs::create_dir_all(out.join("pelicans"))?;
    report::write(&out, &report)?;

    let runner = Run {
        binary: &binary,
        model: &model,
        out: &out,
        options: &options,
        max_ctx: suite.max_ctx,
    };

    for (r, c, t) in jobs {
        let id = format!("r{}-{}-{}", r + 1, cases[t].id, configs[c].id);

        if report["runs"].as_array().unwrap().iter().any(|v| {
            v["id"] == id
                && matches!(
                    v["status"].as_str(),
                    Some("ok" | "incomplete" | "invalid_svg" | "cycling")
                )
        }) {
            continue;
        }

        let record = runner.sample(&configs[c], &cases[t], r)?;
        let failed = record["status"] == "failed";
        let error = record["error"].clone();

        report["runs"]
            .as_array_mut()
            .unwrap()
            .retain(|v| v["id"] != id);
        append(&mut report, "runs", record);
        report::write(&out, &report)?;
        ensure!(
            !failed,
            "{error}; outputs saved in {}; fix and --resume",
            out.display()
        );
    }

    report["utc_finished"] = json!(util::utc()?);

    report::write(&out, &report)?;

    if options.update_readme {
        crate::readme::update(&out)?;
    }

    println!("{}", out.join("gallery.html").display());

    Ok(())
}

fn prepare_report(
    out: &Path,
    options: &Options,
    signature: Value,
    configs: &[Configuration],
    cases: &[Case],
    binary: &Path,
    model: &Path,
) -> Result<Value> {
    if options.resume {
        let mut report = util::json(&out.join("report.json"))?;

        if report["signature"] != signature {
            ensure!(
                suite::only_higher_caps(&report["signature"], &signature),
                "cannot resume: binary, suite, model, or selections changed"
            );
            raise_saved_caps(out, &mut report, signature)?;
        }

        return Ok(report);
    }

    ensure!(
        !out.exists() || fs::read_dir(out)?.next().is_none(),
        "output directory is not empty; choose another or --resume"
    );
    fs::create_dir_all(out)?;

    new_report(signature, configs, cases, binary, model)
}
