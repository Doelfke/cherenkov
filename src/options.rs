use anyhow::{Result, ensure};
use clap::Args;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum PoolBudget {
    #[default]
    Adaptive,
    Max,
    Gb(f64),
}

impl FromStr for PoolBudget {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "adaptive" => return Ok(Self::Adaptive),
            "max" => return Ok(Self::Max),
            _ => {}
        }
        match s.parse::<f64>() {
            Ok(n) if n.is_finite() && n > 0.0 => Ok(Self::Gb(n)),
            _ => Err("pool must be a positive number of decimal GB, adaptive, or max".into()),
        }
    }
}

// TOML keeps its compact number-or-mode syntax; resolved settings use one enum.
impl Serialize for PoolBudget {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Adaptive => serializer.serialize_str("adaptive"),
            Self::Max => serializer.serialize_str("max"),
            Self::Gb(n) => serializer.serialize_f64(*n),
        }
    }
}

impl<'de> Deserialize<'de> for PoolBudget {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PoolVisitor;

        impl<'de> de::Visitor<'de> for PoolVisitor {
            type Value = PoolBudget;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a positive number of decimal GB, 'adaptive', or 'max'")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<PoolBudget, E> {
                value.parse().map_err(E::custom)
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<PoolBudget, E> {
                if value.is_finite() && value > 0.0 {
                    Ok(PoolBudget::Gb(value))
                } else {
                    Err(E::invalid_value(de::Unexpected::Float(value), &self))
                }
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<PoolBudget, E> {
                self.visit_f64(value as f64)
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<PoolBudget, E> {
                self.visit_f64(value as f64)
            }
        }
        deserializer.deserialize_any(PoolVisitor)
    }
}

pub(crate) fn positive(s: &str) -> std::result::Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err("must be a positive integer".into()),
    }
}

pub(crate) fn cut(s: &str) -> std::result::Result<f32, String> {
    match s.parse::<f32>() {
        Ok(n) if n.is_finite() && (0.0..=1.0).contains(&n) => Ok(n),
        _ => Err("weight must be finite and between 0 and 1".into()),
    }
}

/// Measured user choices; developer diagnostics remain environment-only.
#[derive(Args, Debug, Clone)]
pub struct Options {
    /// Expert precision: 4, 3 or 2 bits
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(2..=4))]
    pub experts: u32,
    /// Mid-step fetch precision; follows --experts (mixed mode needs resident 4-bit)
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    pub miss_experts: Option<u32>,
    /// Skip late weak experts; nonzero makes output non-reproducible
    #[arg(long, default_value_t = 0.0, hide_default_value = true, value_parser = cut)]
    pub cut_weak: f32,
    /// Adaptive MTP drafts, 0..3; 0 unloads the draft head
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(0..=3))]
    pub drafts: u8,
    /// Wired expert pool in decimal GB, or max (default: adaptive)
    #[arg(
        long,
        default_value = "adaptive",
        hide_default_value = true,
        value_name = "N|max|adaptive"
    )]
    pub pool_gb: PoolBudget,
    /// Rebuild the selected low-bit store
    #[arg(long)]
    pub repack: bool,
    /// Server policy; CLI runs retain automatic store construction.
    #[arg(skip = true)]
    pub build_missing_store: bool,
    /// Output token limit
    #[arg(long, default_value_t = 64, value_parser = positive)]
    pub max_tokens: usize,
    /// Context capacity; ~22.5 KB/token comes out of the expert pool
    #[arg(long, default_value_t = 2048, value_parser = positive)]
    pub max_ctx: usize,
    /// Skip the chat template
    #[arg(long)]
    pub raw: bool,
    /// Compare GPU rows with the exact CPU reference (slow)
    #[arg(long)]
    pub check: bool,
    /// Repeat using loaded weights, resetting sequence state
    #[arg(long, default_value_t = 1, value_parser = positive)]
    pub repeat: usize,
    /// Ignore EOS for benchmarking
    #[arg(long)]
    pub no_eos: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            experts: 4,
            miss_experts: None,
            cut_weak: 0.0,
            drafts: 2,
            pool_gb: PoolBudget::Adaptive,
            repack: false,
            build_missing_store: true,
            max_tokens: 64,
            max_ctx: 2048,
            raw: false,
            check: false,
            repeat: 1,
            no_eos: false,
        }
    }
}

impl Options {
    pub fn miss_bits(&self) -> u32 {
        self.miss_experts.unwrap_or(self.experts)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=4).contains(&self.experts) && (2..=4).contains(&self.miss_bits()),
            "expert precision must be 4, 3 or 2"
        );
        ensure!(
            self.experts == 4 || self.miss_bits() == self.experts,
            "mixed precision requires --experts 4; only one low-bit store is attached at a time"
        );
        ensure!(
            !self.repack || self.miss_bits() < 4,
            "--repack requires a 2- or 3-bit expert precision"
        );
        ensure!(self.drafts <= 3, "--drafts must be 0..3");
        ensure!(
            self.cut_weak.is_finite() && (0.0..=1.0).contains(&self.cut_weak),
            "invalid cut weight"
        );
        ensure!(
            self.max_tokens > 0 && self.max_ctx > 0 && self.repeat > 0,
            "token limits and repeat must be positive"
        );
        if let PoolBudget::Gb(n) = self.pool_gb {
            ensure!(n.is_finite() && n > 0.0, "invalid pool size");
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/unit/options.rs"]
mod tests;
