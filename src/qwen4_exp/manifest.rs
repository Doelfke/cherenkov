//! On-disk packed tensor layouts, independent of runtime sequence state.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Where a dense tensor lives inside `dense.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub nbytes: u64,
}

/// Expert record layout inside `experts.bin`. One record per (layer, expert):
/// three packed 4-bit weight matrices followed by their scales and biases,
/// padded to a page multiple so each record can be its own Metal buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertLayout {
    pub layers: usize,
    pub experts: usize,
    pub inter: usize,
    pub hidden: usize,
    pub group: usize,
    pub record_bytes: u64,
    pub record_stride: u64,
    /// Tensor prefixes in record order (main layers, then MTP layers).
    pub layer_prefixes: Vec<String>,
    pub gate_w: u64,
    pub up_w: u64,
    pub down_w: u64,
    pub gate_s: u64,
    pub gate_b: u64,
    pub up_s: u64,
    pub up_b: u64,
    pub down_s: u64,
    pub down_b: u64,
}

/// N-gram table layout inside `ngram.bin`: one `row_bytes` record per hashed
/// id holding the packed 4-bit row, then its scales, then its biases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramLayout {
    pub rows: u64,
    pub row_bytes: u64,
    pub dim: usize,
    pub group: usize,
    pub weight_bytes: u64,
    pub scale_bytes: u64,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    pub layer_multipliers: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub source_dir: String,
    pub dense: Vec<DenseEntry>,
    pub experts: ExpertLayout,
    pub ngram: NgramLayout,
}

impl Manifest {
    pub fn load(packed_dir: &Path) -> Result<Self> {
        let path = packed_dir.join("manifest.json");
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes).context("manifest.json")?;

        anyhow::ensure!(
            manifest.version == 1,
            "unsupported packed layout version {}",
            manifest.version
        );

        Ok(manifest)
    }

    pub fn dense(&self, name: &str) -> Result<&DenseEntry> {
        self.dense
            .iter()
            .find(|e| e.name == name)
            .with_context(|| format!("dense tensor {name:?} not in manifest"))
    }
}

pub const PAGE: u64 = 16384;
