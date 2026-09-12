//! Build reports from the shared statistics responses.

mod rows;

use super::Report;
use anyhow::{Result, bail};
use cherenkov::control::{Command, stats as wire};
use rows::*;
use serde_json::Value;
use tabled::{Table, Tabled};

pub(super) enum View {
    Summary(Box<wire::Snapshot<wire::Summary>>),
    Layers(wire::Snapshot<wire::Page<wire::Layer>>),
    Experts(wire::Snapshot<wire::Page<wire::Expert>>),
}

impl View {
    pub fn detail_report(&self, selected: usize) -> Report {
        let mut report = Report::default();

        match self {
            Self::Layers(page) => {
                let Some(layer) = page.data.data.get(selected) else {
                    return report;
                };
                let stats = &layer.streaming;

                report.fields("Layer", &Layer::from(layer));
                report.fields("Prediction", &Prediction::from(&stats.prediction));
                report.table(
                    "Expert precision",
                    Table::new(stats.quant.iter().map(Quant::from)),
                );
                report.fields("Expert reads", &Reads::from(&stats.reads()));
                report.fields("CPU phases", &Cpu::from(&stats.phases));
                report.fields("GPU phases", &Gpu::from(&stats.phases));
            }
            Self::Experts(page) => {
                if let Some(expert) = page.data.data.get(selected) {
                    report.fields("Expert", &Expert::from(expert));
                }
            }
            Self::Summary(_) => return self.report(),
        }

        report
    }

    pub fn from_response(command: &Command, response: &Value) -> Result<Self> {
        let response = response.clone();

        Ok(match command {
            Command::StatsSummary => Self::Summary(serde_json::from_value(response)?),
            Command::StatsLayers { .. } => Self::Layers(serde_json::from_value(response)?),
            Command::StatsExperts { .. } => Self::Experts(serde_json::from_value(response)?),
            _ => bail!("not a statistics command"),
        })
    }

    pub fn report(&self) -> Report {
        let report = Report::default();

        match self {
            Self::Summary(summary) => summary_report(summary, report),
            Self::Layers(page) => page_report(page, "Layers", Layer::from, report),
            Self::Experts(page) => page_report(page, "Experts", Expert::from, report),
        }
    }
}

fn summary_report(snapshot: &wire::Snapshot<wire::Summary>, mut report: Report) -> Report {
    let summary = &snapshot.data;

    report.fields("Observation", &Observation::from(&snapshot.observation));
    report.fields("Rates since engine load", &Rates::from(&summary.rates));
    report.fields("Prefill chunks", &Prefill::from(&summary.prefill));
    report.fields("Expert reads", &Reads::from(&summary.reads));
    report.fields("CPU phases", &Cpu::from(&summary.streaming.phases));
    report.fields("GPU phases", &Gpu::from(&summary.streaming.phases));
    report.fields(
        "Prediction",
        &Prediction::from(&summary.streaming.prediction),
    );
    report.table(
        "Expert precision",
        Table::new(summary.streaming.quant.iter().map(Quant::from)),
    );

    report
}

fn page_report<'a, T, R: Tabled>(
    snapshot: &'a wire::Snapshot<wire::Page<T>>,
    title: &str,
    row: impl Fn(&'a T) -> R,
    mut report: Report,
) -> Report {
    let page = &snapshot.data;

    report.fields("Observation", &Observation::from(&snapshot.observation));
    report.table(title, Table::new(page.data.iter().map(row)));
    report.note(format!(
        "{} entries; offset {}; total {}.",
        page.data.len(),
        page.offset,
        page.total
    ));

    if let Some(offset) = page.next_offset {
        report.note(format!("Next page: --offset {offset}."));
    }

    report
}
