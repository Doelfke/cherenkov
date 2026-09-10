//! Static README tables and images, generated from a saved benchmark report.
use crate::{report, util};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{fmt::Write as _, fs, path::Path};

const START: &str = "<!-- benchmarks:start -->";
const END: &str = "<!-- benchmarks:end -->";

fn cell(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace(['\n', '\r'], " ")
}

fn link(path: &Path) -> String {
    path.to_string_lossy()
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('(', "%28")
        .replace(')', "%29")
}

fn label(config: &Value) -> &str {
    if config["id"] == "misses-2bit" {
        return "4-bit / 2-bit misses + cut";
    }

    report::text(&config["label"])
}

fn table(data: &Value, rows: &[Value], kind: &str, metric: &str) -> Result<String> {
    let cases: Vec<_> = data["cases"]
        .as_array()
        .context("report cases")?
        .iter()
        .filter(|case| case["kind"].as_str().unwrap_or("decode") == kind)
        .collect();

    if cases.is_empty() {
        return Ok(String::new());
    }

    let mut text = String::from("| Experts |");

    for case in &cases {
        write!(text, " {} {metric} |", cell(report::text(&case["id"])))?;
    }

    text.push_str("\n| --- |");
    text.push_str(&" ---: |".repeat(cases.len()));
    text.push('\n');

    for config in data["configurations"]
        .as_array()
        .context("configurations")?
    {
        write!(text, "| {} |", cell(label(config)))?;

        for case in &cases {
            let row = rows
                .iter()
                .find(|r| r["case"] == case["id"] && r["configuration"] == config["id"]);
            let key = if kind == "prefill" { "pp_s" } else { "tg_s" };

            match row.and_then(|r| r[key].as_f64()) {
                Some(rate) => write!(text, " {rate:.2} |")?,
                None => text.push_str(" n/a |"),
            }
        }

        text.push('\n');
    }

    text.push('\n');

    Ok(text)
}

fn drawings(data: &Value, directory: &Path) -> Result<String> {
    let runs = data["runs"].as_array().context("runs")?;
    let mut images = Vec::new();

    for config in data["configurations"]
        .as_array()
        .context("configurations")?
    {
        let run = runs.iter().rev().find(|r| {
            r["configuration"] == config["id"] && r["status"] == "ok" && r["svg"].is_string()
        });
        let Some(run) = run else {
            continue;
        };
        let title = cell(label(config));
        let path = link(&directory.join(run["svg"].as_str().unwrap()));

        images.push((title, format!("![Pelican]({path})")));
    }

    let mut text = String::new();

    if !images.is_empty() {
        text.push_str("### Pelicans\n\nUnedited model outputs from the same run.\n\n");
    }

    for pair in images.chunks(2) {
        writeln!(
            text,
            "| {} |",
            pair.iter()
                .map(|p| p.0.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        )?;
        writeln!(text, "| {} |", vec!["---"; pair.len()].join(" | "))?;
        writeln!(
            text,
            "| {} |\n",
            pair.iter()
                .map(|p| p.1.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        )?;
    }

    Ok(text)
}

pub fn render(data: &Value, directory: &Path) -> Result<String> {
    ensure!(
        data["utc_finished"].is_string(),
        "finish the benchmark before updating the README"
    );

    let rows = report::rows(data)?;

    ensure!(!rows.is_empty(), "no valid benchmark samples");

    let runs = data["runs"].as_array().context("runs")?;
    let valid = runs.iter().filter(|r| r["status"] == "ok").count();
    let provenance = &data["provenance"];
    let commit = report::text(&provenance["commit"]);
    let memory = provenance["memory_bytes"]
        .as_str()
        .and_then(|v| v.parse::<f64>().ok())
        .context("machine memory")?
        / 1073741824.0;
    let metal = rows
        .iter()
        .filter_map(|r| r["metal_gb"].as_f64())
        .fold(0.0, f64::max);
    let report_link = link(&directory.join("report.json"));
    let gallery_link = link(&directory.join("gallery.html"));
    let mut text = format!(
        "Measured on {} with {memory:.0} GiB memory; {metal:.2} GB reported Metal allocation.\nSaved revision `{}`; {valid} valid samples.\n\nLoading and store construction are excluded. Answer lengths vary;\ncompare completion times in the full report.\n\n[Full report]({report_link}).\n\n",
        cell(report::text(&provenance["hardware"])),
        cell(commit.get(..7).unwrap_or(commit)),
    );

    text.push_str(&table(data, &rows, "decode", "tg/s")?);
    text.push_str(&table(data, &rows, "prefill", "pp/s")?);
    text.push_str("Lower precision changes outputs. Measurements belong to the saved revision;\nsee current validation for subsequent changes. Metal allocation is reported\nafter prefill scratch release, not at its transient peak.\n\n");
    append_notes(&mut text, data)?;
    writeln!(text, "[All timings and outputs]({gallery_link}).\n")?;
    text.push_str(&drawings(data, directory)?);

    Ok(text)
}

fn append_notes(text: &mut String, data: &Value) -> Result<()> {
    let configs = data["configurations"]
        .as_array()
        .context("configurations")?;
    let cut = configs
        .iter()
        .filter_map(|c| c["args"].as_array())
        .any(|args| {
            args.windows(2).any(|pair| {
                pair[0] == "--cut-weak"
                    && pair[1]
                        .as_str()
                        .and_then(|v| v.parse::<f32>().ok())
                        .is_some_and(|v| v > 0.0)
            })
        });

    if cut {
        text.push_str("Settings with a deadline cut are not reproducible.\n\n");
    }

    if let Some(notes) = data["readme_notes"].as_array() {
        for note in notes {
            writeln!(text, "{}\n", report::text(note))?;
        }
    }

    Ok(())
}

pub fn replace_section(readme: &str, generated: &str) -> Result<String> {
    ensure!(
        readme.matches(START).count() == 1 && readme.matches(END).count() == 1,
        "README needs exactly one benchmark marker pair"
    );

    let start = readme.find(START).unwrap() + START.len();
    let end = readme.find(END).unwrap();

    ensure!(start < end, "README benchmark markers are reversed");

    Ok(format!(
        "{}\n\n{}\n{}",
        &readme[..start],
        generated.trim_end(),
        &readme[end..]
    ))
}

pub fn update(directory: &Path) -> Result<()> {
    let root = util::root().canonicalize()?;
    let directory = directory.canonicalize()?;
    let relative = directory
        .strip_prefix(&root)
        .context("README results must be inside the repository")?;
    let data = util::json(&directory.join("report.json"))?;

    for run in data["runs"].as_array().context("runs")? {
        if let Some(svg) = run["svg"].as_str() {
            ensure!(
                directory.join(svg).is_file(),
                "missing pelican image: {svg}"
            );
        }
    }

    let generated = render(&data, relative)?;
    let path = root.join("README.md");
    let text = replace_section(&fs::read_to_string(&path)?, &generated)?;

    fs::write(&path, text)?;
    println!("Updated {} from {}", path.display(), directory.display());

    Ok(())
}
