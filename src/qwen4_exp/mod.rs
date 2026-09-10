//! qwen4-exp (model_type qwen4_exp_text): 48-layer hybrid of Gated
//! DeltaNet and sparse attention, 512-expert MoE per layer, four-stream
//! gated residual ("hyper connections"), a hashed n-gram embedding injected
//! at one layer, and a one-layer MTP head. 125B parameters, 6B active.
//!
//! Weights come from the packed layout written by `pack`. Metal wires whole buffers, so
//! every expert record must be its own page-aligned buffer, and the MLX
//! shards are not even 4-byte aligned on disk.

pub mod cpu;
pub mod gpu;
pub mod lowbit;
pub mod pack;
pub mod packed;

mod config;
mod manifest;
pub use config::{Qwen4ExpConfig, RopeParams};
pub use manifest::{DenseEntry, ExpertLayout, Manifest, NgramLayout, PAGE};
