//! Markdown shared by the README and saved benchmark reports.

use crate::report;
use anyhow::{Context, Result};
use serde_json::Value;
use std::{fmt::Write as _, path::Path};

pub(crate) fn cell(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace(['\n', '\r'], " ")
}

pub(crate) fn link(path: &Path) -> String {
    path.to_string_lossy()
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F")
        .replace('(', "%28")
        .replace(')', "%29")
}

pub(crate) fn answers(data: &Value) -> Result<String> {
    let mut rows = String::new();

    for run in data["runs"].as_array().context("runs")? {
        let Some(output) = run["output"].as_str() else {
            continue;
        };

        writeln!(
            rows,
            "| [{}]({}) | {} |",
            cell(report::text(&run["id"])),
            link(Path::new(output)),
            cell(report::text(&run["status"]))
        )?;
    }

    if rows.is_empty() {
        return Ok(rows);
    }

    Ok(format!(
        "\n## Answers\n\n| Sample | Status |\n| --- | --- |\n{rows}"
    ))
}

pub(crate) fn label(config: &Value) -> &str {
    if config["id"] == "misses-2bit" {
        return "4-bit / 2-bit misses + cut";
    }

    report::text(&config["label"])
}

pub(crate) fn drawings(data: &Value, directory: &Path) -> Result<String> {
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
