//! Small, bounded snapshots shared with the control thread. No prompts or logits.

use crate::config::{Config, Source};
use anyhow::{Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::Mutex;
use std::time::Instant;

pub struct State {
    source: Source,
    config: Mutex<Versioned>,
    stats: Mutex<Stats>,
    started: Instant,
}

#[derive(Clone, Serialize)]
pub struct Versioned {
    pub generation: u64,
    pub config: Config,
}

#[derive(Default, Serialize)]
pub struct Stats {
    pub ready: bool,
    pub queued_requests: usize,
    pub active_requests: usize,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub rejected_requests: u64,
    pub generated_tokens: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub current: Option<Current>,
    pub cache: CacheStats,
    pub memory: MemoryStats,
    pub http_address: Option<String>,
}

#[derive(Serialize)]
pub struct Current {
    pub config_generation: u64,
    pub phase: &'static str,
    pub generated_tokens: u64,
}

#[derive(Default, Serialize)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub evictions: u64,
}

#[derive(Default, Serialize)]
pub struct MemoryStats {
    pub metal_allocated_bytes_observed: u64,
    pub expert_pool_bytes: usize,
    pub resident_experts: usize,
    pub observed_at_uptime_seconds: f64,
}

impl State {
    pub fn new(source: Source, config: Config) -> Self {
        Self {
            source,
            config: Mutex::new(Versioned {
                generation: 1,
                config,
            }),
            stats: Mutex::new(Stats::default()),
            started: Instant::now(),
        }
    }

    pub fn config(&self) -> Versioned {
        self.config.lock().unwrap().clone()
    }

    pub fn show_config(&self) -> Value {
        json!({"effective": self.config(), "source": self.source.path,
            "reloadable_sections": ["defaults"], "restart_required_sections": ["server", "limits", "experts"]})
    }

    pub fn reload(&self) -> Result<Value> {
        ensure!(
            self.source.path.is_some(),
            "server was started without --config"
        );
        let next = self.source.resolve()?;
        let mut current = self.config.lock().unwrap();
        let changes = current.config.restart_changes(&next);
        ensure!(
            changes.is_empty(),
            "restart required for changed sections: {}",
            changes.join(", ")
        );
        if current.config != next {
            current.config = next;
            current.generation += 1;
        }
        Ok(json!({"generation": current.generation, "config": current.config}))
    }

    pub fn update(&self, f: impl FnOnce(&mut Stats)) {
        f(&mut self.stats.lock().unwrap());
    }

    pub fn status(&self) -> Value {
        json!({"uptime_seconds": self.started.elapsed().as_secs_f64(),
            "stats": *self.stats.lock().unwrap(),
            "capabilities": {"active_sequences": 1, "sampling": false, "retained_sessions": false}})
    }

    pub fn begin(&self, generation: u64) {
        self.update(|s| {
            s.queued_requests -= 1;
            s.active_requests = 1;
            s.current = Some(Current {
                config_generation: generation,
                phase: "validating",
                generated_tokens: 0,
            });
        });
    }

    pub fn finish(&self, success: bool) {
        self.update(|s| {
            s.active_requests = 0;
            s.current = None;
            if success {
                s.completed_requests += 1;
            } else {
                s.failed_requests += 1;
            }
        });
    }

    pub fn phase(&self, phase: &'static str) {
        self.update(|s| {
            if let Some(current) = &mut s.current {
                current.phase = phase;
            }
        });
    }

    pub fn token(&self) {
        self.update(|s| {
            s.generated_tokens += 1;
            if let Some(current) = &mut s.current {
                current.generated_tokens += 1;
            }
        });
    }

    pub(crate) fn observe(
        &self,
        gpu: &crate::qwen4_exp::gpu::Gpu<'_>,
        cache: &crate::prefix_cache::PrefixCache,
    ) {
        let (entries, bytes, evictions) = cache.stats();
        self.update(|s| {
            s.cache = CacheStats {
                entries,
                bytes,
                evictions,
            };
            s.memory = MemoryStats {
                metal_allocated_bytes_observed: (gpu.allocated_gb() * 1e9) as u64,
                expert_pool_bytes: gpu.pool_bytes(),
                resident_experts: gpu.pool_resident(),
                observed_at_uptime_seconds: self.started.elapsed().as_secs_f64(),
            };
        });
    }
}
