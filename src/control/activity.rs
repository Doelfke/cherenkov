//! Paged views of the worker's copied expert counters; no GPU access.

use super::stats::{Expert, Layer, Page, Rates, Summary};
use crate::qwen4_exp::gpu::{ExpertActivity, ExpertCounters, LayerStats};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::ops::Range;

fn range(total: usize, offset: usize, limit: usize) -> Result<Range<usize>> {
    ensure!(
        (1..=128).contains(&limit),
        "limit must be between 1 and 128"
    );
    ensure!(offset <= total, "offset exceeds available entries");

    Ok(offset..offset.saturating_add(limit).min(total))
}

fn page<T: Serialize>(total: usize, range: Range<usize>, data: Vec<T>) -> Result<Page<T>> {
    // Leave room for framing and observation metadata under the 64 KiB limit.
    let mut bytes = 0;
    let mut bounded = Vec::new();

    for entry in data {
        bytes += serde_json::to_vec(&entry)?.len() + 1;

        if bytes > 60 * crate::units::BYTES_PER_KIB {
            break;
        }

        bounded.push(entry);
    }

    ensure!(
        range.is_empty() || !bounded.is_empty(),
        "one statistics entry exceeds the control frame"
    );

    let end = range.start + bounded.len();

    Ok(Page {
        offset: range.start,
        next_offset: (end < total).then_some(end),
        total,
        data: bounded,
    })
}

pub(super) fn layers(
    activity: &ExpertActivity,
    offset: usize,
    limit: usize,
) -> Result<Page<Layer>> {
    let total = activity.layers.len();
    let range = range(total, offset, limit)?;
    let data = range
        .clone()
        .map(|layer| {
            let start = layer * activity.experts_per_layer;
            let mut counters = ExpertCounters::default();

            for &record in &activity.records[start..start + activity.experts_per_layer] {
                counters.add(record);
            }

            Layer {
                layer,
                name: activity.layer_prefixes[layer].clone(),
                experts: activity.experts_per_layer,
                counters,
                streaming: activity.layers[layer].clone(),
            }
        })
        .collect();

    page(total, range, data)
}

pub(super) fn summary(activity: &ExpertActivity) -> Summary {
    let mut total = LayerStats::default();

    for layer in &activity.layers {
        total.add(layer);
    }

    let reads = total.reads();
    let phase = total.phases;
    let gaps = phase.router_to_resident_seconds + phase.resident_to_fetched_seconds;
    let stages = phase.resident_seconds + phase.fetched_stage_seconds;

    Summary {
        streaming: total,
        reads,
        rates: Rates {
            bytes_per_second: ratio(reads.completed_bytes as f64, activity.elapsed_seconds),
            reads_per_second: ratio(reads.completed_reads as f64, activity.elapsed_seconds),
            gpu_handoff_gap_fraction: ratio(gaps, gaps + stages),
            gpu_stage_fraction: ratio(stages, gaps + stages),
        },
    }
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| numerator / denominator)
}

pub(super) fn experts(
    activity: &ExpertActivity,
    layer: usize,
    offset: usize,
    limit: usize,
) -> Result<Page<Expert>> {
    ensure!(layer < activity.layers.len(), "unknown expert layer");

    let total = activity.experts_per_layer;
    let range = range(total, offset, limit)?;
    let data = range
        .clone()
        .map(|expert| Expert {
            layer,
            expert,
            counters: activity.records[layer * total + expert],
        })
        .collect();

    page(total, range, data)
}
