use anyhow::{Context, Result, ensure};
use regex::Regex;
use serde_json::{Value, json};

const BYTES_PER_GB: f64 = 1_000_000_000.0;

fn fields(text: &str, label: &str, pattern: &str) -> Result<Vec<f64>> {
    let regex = Regex::new(pattern)?;
    let captures = regex
        .captures(text)
        .with_context(|| format!("incomplete engine telemetry: {label}"))?;

    captures
        .iter()
        .skip(1)
        .map(|m| Ok(m.unwrap().as_str().parse()?))
        .collect()
}

pub fn parse(text: &str) -> Result<Value> {
    let load = fields(text, "load", r"load ([\d.]+)s, clock probe ([\d.]+) ms")?;
    let prefill = fields(
        text,
        "prefill",
        r"prefill (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)",
    )?;
    let decode = fields(
        text,
        "decode",
        r"decode (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\).*?\| (\d+) steps, mean step ([\d.]+) ms.*?gpu-(?:active|span) ([\d.]+) ms.*?io wait ([\d.]+) ms",
    )?;
    let memory = memory_stats(text)?;
    let metal_gb = match &memory {
        Some(stats) => {
            stats["metal_allocated_bytes_observed"]
                .as_u64()
                .context("memory_stats: expected integer Metal bytes")? as f64
                / BYTES_PER_GB
        }
        None => fields(text, "memory", r"\| ([\d.]+) GB Metal")?[0],
    };
    let clock = fields(text, "clock_end", r"clock probe at end ([\d.]+) ms")?;
    let drafts = fields(text, "drafts", r"drafts (\d+), accepted ([\d.]+)/step")?;
    let build = fields(
        text,
        "store_build",
        r"built the [23]-bit store in ([\d.]+)s",
    )
    .ok();

    Ok(json!({
        "store_build_seconds":build.map(|v|v[0]), "load_seconds":load[0], "clock_start_ms":load[1],
        "prompt_tokens":prefill[0] as u64, "prefill_seconds":prefill[1], "pp_s":prefill[2],
        "output_tokens":decode[0] as u64, "decode_seconds":decode[1], "tg_s":decode[2],
        "steps":decode[3] as u64, "mean_step_ms":decode[4], "gpu_span_ms":decode[5], "io_wait_ms":decode[6],
        "metal_gb":metal_gb, "memory":memory,
        "clock_end_ms":clock[0], "drafts":drafts[0] as u64, "accepted_drafts_per_step":drafts[1]
    }))
}

fn memory_stats(text: &str) -> Result<Option<Value>> {
    let mut snapshots = text
        .lines()
        .filter_map(|line| line.strip_prefix("memory_stats "));
    let Some(line) = snapshots.next() else {
        return Ok(None);
    };

    ensure!(
        snapshots.next().is_none(),
        "multiple memory_stats snapshots in one run"
    );

    let value: Value = serde_json::from_str(line).context("invalid memory_stats JSON")?;

    ensure!(value.is_object(), "memory_stats must be an object");

    Ok(Some(value))
}

pub fn extract_svg(text: &str) -> Result<(&str, usize)> {
    let regex = Regex::new(r"(?is)<svg\b.*?</svg\s*>")?;
    let svg = regex
        .find(text)
        .context("no complete SVG: output may have hit its token limit")?
        .as_str();
    let document = roxmltree::Document::parse(svg)?;

    anyhow::ensure!(
        document.root_element().tag_name().name() == "svg",
        "root element is not svg"
    );

    Ok((
        svg,
        document.descendants().filter(|n| n.is_element()).count(),
    ))
}

pub fn repeated_tail(text: &str) -> Option<Value> {
    let mut words: Vec<&str> = text.split_whitespace().rev().skip(1).take(4096).collect();

    words.reverse();

    for period in 32..=512.min(words.len() / 4) {
        let tail = &words[words.len() - 4 * period..];
        let block = &words[words.len() - period..];

        if tail.chunks_exact(period).all(|chunk| chunk == block) {
            return Some(
                json!({"words_per_cycle":period,"repetitions":4,"example":block.join(" ").chars().take(240).collect::<String>()}),
            );
        }
    }

    None
}
