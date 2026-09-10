//! Model configuration for the qwen4_exp text engine.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Text-model configuration, parsed from config.json's `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen4ExpConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub partial_rotary_factor: f64,
    pub rope_parameters: RopeParams,
    #[serde(default = "four")]
    pub hc_count: usize,
    #[serde(default = "d320")]
    pub hc_lowrank: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "yes")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default = "four")]
    pub ple_conv_kernel_size: usize,
    pub ple_embed_dim: usize,
    #[serde(default = "three")]
    pub ngram_size: usize,
    #[serde(default = "eight")]
    pub heads_per_ngram: usize,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    #[serde(default = "silu")]
    pub output_gate_type: String,
    pub eos_token_id: u32,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
}

fn four() -> usize {
    4
}

fn three() -> usize {
    3
}

fn eight() -> usize {
    8
}

fn d320() -> usize {
    320
}

fn yes() -> bool {
    true
}

fn silu() -> String {
    "silu".into()
}

impl Qwen4ExpConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let root: serde_json::Value = serde_json::from_slice(&bytes).context("config.json")?;
        let model_type = root["model_type"].as_str().context("model_type missing")?;

        anyhow::ensure!(
            matches!(model_type, "qwen4_exp" | "qwen4_exp_text"),
            "unsupported model_type {model_type}; expected qwen4_exp"
        );

        let text = root.get("text_config").cloned().unwrap_or(root);

        anyhow::ensure!(
            text["model_type"] == "qwen4_exp_text",
            "expected qwen4_exp_text configuration"
        );

        let cfg: Qwen4ExpConfig = serde_json::from_value(text).context("text_config")?;

        anyhow::ensure!(
            cfg.layer_types.len() == cfg.num_hidden_layers,
            "layer_types length mismatch"
        );
        anyhow::ensure!(cfg.hc_count > 1, "hc_count must exceed 1");

        Ok(cfg)
    }

    pub fn is_linear(&self, layer: usize) -> bool {
        self.layer_types[layer] == "linear_attention"
    }

    /// Zero-based layer index carrying the n-gram (PLE) block, if any.
    pub fn ple_layer(&self) -> Option<usize> {
        self.ple_layer_ids.first().map(|id| id - 1)
    }

    pub fn hc_hidden(&self) -> usize {
        self.hc_count * self.hidden_size
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4_exp/config.rs"]
mod tests;
