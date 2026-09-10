//! Server configuration: built-ins, one TOML file, then explicit CLI overrides.

use crate::options::{Options, PoolBudget, cut, positive};
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};

pub fn default_socket() -> PathBuf {
    std::env::temp_dir()
        .join(format!("cherenkov-{}", unsafe { libc::geteuid() }))
        .join("control.sock")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: Server,
    pub limits: Limits,
    pub defaults: Defaults,
    pub experts: Experts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Server {
    pub model_dir: Option<PathBuf>,
    /// Optional portable root for model data, scratch and configuration.
    pub root: Option<PathBuf>,
    pub port: u16,
    pub socket: PathBuf,
    pub drafts: u8,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            model_dir: None,
            root: None,
            port: 8080,
            socket: default_socket(),
            drafts: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    /// Metal allocations plus reserved prefix cache, not whole-process RSS.
    pub memory_gb: f64,
    pub context_tokens: usize,
    pub prefix_cache_mib: usize,
    pub cache_max_entries: usize,
    /// Zero disables time-based expiry; the byte and entry limits still apply.
    pub cache_idle_seconds: u64,
    pub queued_requests: usize,
    pub http_readers: usize,
    pub request_bytes: usize,
    pub max_output_tokens: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            memory_gb: 25.0,
            context_tokens: 2048,
            prefix_cache_mib: 512,
            cache_max_entries: 16,
            cache_idle_seconds: 900,
            queued_requests: 8,
            http_readers: 32,
            request_bytes: 4 * 1024 * 1024,
            max_output_tokens: 262144,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Defaults {
    pub max_tokens: usize,
    pub no_eos: bool,
    pub stream: bool,
    pub include_usage: bool,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            no_eos: false,
            stream: false,
            include_usage: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Experts {
    pub resident_bits: u32,
    /// None follows resident_bits; mixed mode requires four-bit residents.
    pub miss_bits: Option<u32>,
    pub pool_gb: PoolBudget,
    pub cut_weak: f32,
    pub build_missing_store: bool,
}

impl Default for Experts {
    fn default() -> Self {
        Self {
            resident_bits: 4,
            miss_bits: None,
            pool_gb: PoolBudget::Adaptive,
            cut_weak: 0.0,
            build_missing_store: true,
        }
    }
}

impl Config {
    pub fn paths(&self) -> Result<crate::storage::Paths> {
        crate::storage::Paths::new(self.server.root.as_deref())
    }

    pub fn model_dir(&self) -> Result<PathBuf> {
        Ok(match &self.server.model_dir {
            Some(dir) => dir.clone(),
            None => self.paths()?.default_model(),
        })
    }

    pub fn options(&self) -> Options {
        Options {
            experts: self.experts.resident_bits,
            miss_experts: self.experts.miss_bits,
            pool_gb: self.experts.pool_gb,
            cut_weak: self.experts.cut_weak,
            drafts: self.server.drafts,
            max_ctx: self.limits.context_tokens,
            max_tokens: self.defaults.max_tokens,
            no_eos: self.defaults.no_eos,
            build_missing_store: self.experts.build_missing_store,
            ..Options::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.options().validate()?;
        let l = &self.limits;
        ensure!(
            l.memory_gb.is_finite() && l.memory_gb > 0.0 && l.memory_gb <= 25.0,
            "limits.memory_gb must be positive and at most 25 decimal GB"
        );
        ensure!(
            l.context_tokens <= 262144,
            "context exceeds the model's 262144-token limit"
        );
        ensure!(
            l.prefix_cache_mib <= 2048,
            "prefix cache must be at most 2048 MiB"
        );
        ensure!(
            (1..=1024).contains(&l.cache_max_entries),
            "cache_max_entries must be 1..1024"
        );
        ensure!(
            self.cache_bytes() < self.memory_bytes(),
            "prefix cache exceeds memory budget"
        );
        ensure!(
            (1..=1024).contains(&l.queued_requests),
            "queued_requests must be 1..1024"
        );
        ensure!(
            (1..=256).contains(&l.http_readers),
            "http_readers must be 1..256"
        );
        ensure!(
            (1..=16 * 1024 * 1024).contains(&l.request_bytes),
            "request_bytes must be 1..16777216"
        );
        ensure!(
            (1..=262144).contains(&l.max_output_tokens),
            "max_output_tokens must be 1..262144"
        );
        ensure!(
            self.defaults.max_tokens <= l.max_output_tokens,
            "default max_tokens exceeds output policy"
        );
        ensure!(
            self.defaults
                .max_tokens
                .checked_add(self.server.drafts as usize)
                .is_some_and(|n| n < l.context_tokens),
            "default output and drafts leave no prompt capacity"
        );
        ensure!(
            self.server.socket.is_absolute(),
            "control socket path must be absolute"
        );
        Ok(())
    }

    pub fn cache_bytes(&self) -> usize {
        self.limits.prefix_cache_mib * 1024 * 1024
    }

    pub fn memory_bytes(&self) -> usize {
        (self.limits.memory_gb * 1e9) as usize
    }

    pub fn restart_changes(&self, new: &Self) -> Vec<&'static str> {
        let mut changes = Vec::new();
        if self.server != new.server {
            changes.push("server");
        }
        if self.limits != new.limits {
            changes.push("limits");
        }
        if self.experts != new.experts {
            changes.push("experts");
        }
        changes
    }
}

/// Only explicit CLI values are retained and reapplied on every reload.
/// Unlike resolved engine options, absent fields never supply a default.
#[derive(Args, Debug, Clone, Default)]
pub struct Overrides {
    #[arg(skip)]
    pub root: Option<PathBuf>,
    pub model_dir: Option<PathBuf>,
    /// HTTP port on localhost (default: 8080)
    #[arg(long)]
    pub port: Option<u16>,
    /// Control socket (default: private per-user temporary directory)
    #[arg(long)]
    pub socket: Option<PathBuf>,
    /// Prefix cache in MiB; 0 disables it
    #[arg(long = "prefix-cache-mb")]
    pub prefix_cache_mib: Option<usize>,
    /// Expert precision: 4, 3 or 2 bits (default: 4)
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    pub experts: Option<u32>,
    /// Mid-step fetch precision; follows --experts (mixed mode needs resident 4-bit)
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=4))]
    pub miss_experts: Option<u32>,
    /// Skip late weak experts; nonzero makes output non-reproducible
    #[arg(long, value_parser = cut)]
    pub cut_weak: Option<f32>,
    /// Adaptive MTP drafts, 0..3 (default: 2); 0 unloads the draft head
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=3))]
    pub drafts: Option<u8>,
    /// Wired expert pool in decimal GB, or max (default: adaptive)
    #[arg(long, value_name = "N|max|adaptive")]
    pub pool_gb: Option<PoolBudget>,
    /// Rebuild the selected low-bit store at startup
    #[arg(long)]
    pub repack: bool,
    /// Default output token limit (default: 64)
    #[arg(long, value_parser = positive)]
    pub max_tokens: Option<usize>,
    /// Context capacity; ~22.5 KB/token comes out of the expert pool (default: 2048)
    #[arg(long, value_parser = positive)]
    pub max_ctx: Option<usize>,
    /// Ignore EOS for benchmarking; =false overrides a configured true
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
    pub no_eos: Option<bool>,
}

impl Overrides {
    fn apply(&self, c: &mut Config) {
        self.apply_server(&mut c.server);
        self.apply_experts(&mut c.experts);
        if let Some(n) = self.prefix_cache_mib {
            c.limits.prefix_cache_mib = n;
        }
        if let Some(n) = self.max_ctx {
            c.limits.context_tokens = n;
        }
        if let Some(n) = self.max_tokens {
            c.defaults.max_tokens = n;
        }
        if let Some(no_eos) = self.no_eos {
            c.defaults.no_eos = no_eos;
        }
    }

    fn apply_server(&self, server: &mut Server) {
        if let Some(root) = &self.root {
            server.root = Some(root.clone());
        }
        if let Some(p) = &self.model_dir {
            server.model_dir = Some(p.clone());
        }
        if let Some(p) = self.port {
            server.port = p;
        }
        if let Some(p) = &self.socket {
            server.socket = p.clone();
        }
        if let Some(n) = self.drafts {
            server.drafts = n;
        }
    }

    fn apply_experts(&self, experts: &mut Experts) {
        if let Some(n) = self.experts {
            experts.resident_bits = n;
        }
        if let Some(n) = self.miss_experts {
            experts.miss_bits = Some(n);
        }
        if let Some(n) = self.cut_weak {
            experts.cut_weak = n;
        }
        if let Some(pool) = self.pool_gb {
            experts.pool_gb = pool;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Source {
    pub path: Option<PathBuf>,
    pub overrides: Overrides,
}

impl Source {
    pub fn resolve(&self) -> Result<Config> {
        let mut c = if let Some(path) = &self.path {
            let file =
                std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            let mut s = String::new();
            file.take(65537).read_to_string(&mut s)?;
            ensure!(s.len() <= 65536, "config exceeds 64 KiB");
            let mut c: Config =
                toml::from_str(&s).with_context(|| format!("parsing {}", path.display()))?;
            let base = path.parent().context("config path has no parent")?;
            if let Some(p) = &mut c.server.model_dir
                && p.is_relative()
            {
                *p = base.join(&*p);
            }
            if let Some(root) = &mut c.server.root
                && root.is_relative()
            {
                *root = base.join(&*root);
            }
            if c.server.socket.is_relative() {
                c.server.socket = base.join(&c.server.socket);
            }
            c
        } else {
            Config::default()
        };
        self.overrides.apply(&mut c);
        c.validate()?;
        Ok(c)
    }
}

pub fn absolute(path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}

#[cfg(test)]
#[path = "../tests/unit/config.rs"]
mod tests;
