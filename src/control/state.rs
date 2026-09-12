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
    pub cancelled_requests: u64,
    pub active_state_reserved_bytes: usize,
    /// Contended prefill chunk size the worker is currently pacing toward.
    pub prefill_chunk_tokens: usize,
    pub sessions: SessionStats,
    pub active: Vec<ActiveRequest>,
    #[serde(skip)]
    pub(crate) activity: crate::qwen4_exp::gpu::ExpertActivity,
}

#[derive(Default, Serialize)]
pub struct SessionStats {
    pub entries: usize,
    pub bytes: usize,
    pub evictions: u64,
}

#[derive(Serialize)]
pub struct ActiveRequest {
    pub id: String,
    pub session_id: Option<String>,
    pub phase: &'static str,
    #[serde(flatten)]
    pub usage: crate::server::UsageStats,
    pub reserved_state_bytes: usize,
}

#[derive(Serialize)]
pub struct Current {
    pub request_id: String,
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
    #[serde(flatten)]
    pub resources: crate::qwen4_exp::gpu::MemoryStats,
    pub observed_at_uptime_seconds: f64,
}

impl State {
    pub(super) fn summary(&self) -> Result<Value> {
        self.activity_page(|activity| Ok(super::activity::summary(activity)))
    }
    pub(super) fn layers(&self, offset: usize, limit: usize) -> Result<Value> {
        self.activity_page(|activity| super::activity::layers(activity, offset, limit))
    }

    pub(super) fn experts(&self, layer: usize, offset: usize, limit: usize) -> Result<Value> {
        self.activity_page(|activity| super::activity::experts(activity, layer, offset, limit))
    }

    fn activity_page<T: Serialize>(
        &self,
        query: impl FnOnce(&crate::qwen4_exp::gpu::ExpertActivity) -> Result<T>,
    ) -> Result<Value> {
        let stats = self.stats.lock().unwrap();

        ensure!(stats.ready, "model not ready: still loading");
        stats.activity.validate_dimensions()?;

        let snapshot = super::stats::Snapshot {
            observation: super::stats::Observation {
                observed_at_uptime_seconds: stats.memory.observed_at_uptime_seconds,
                elapsed_seconds: stats.activity.elapsed_seconds,
                gpu_timestamps_available: stats.activity.gpu_timestamps_available,
                gpu_timing: stats.activity.gpu_timing.clone(),
            },
            data: query(&stats.activity)?,
        };

        drop(stats);

        Ok(serde_json::to_value(snapshot)?)
    }

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
        let config = self.config();

        json!({"uptime_seconds": self.started.elapsed().as_secs_f64(),
            "stats": *self.stats.lock().unwrap(),
            "capabilities": {"active_sequences": config.config.limits.active_requests,
                "sampling": true, "retained_sessions": config.config.limits.max_sessions > 0, "cancellation": true}})
    }

    pub fn begin(&self) {
        self.update(|s| {
            s.queued_requests -= 1;
            s.active_requests += 1;
        });
    }

    pub fn current(&self, request_id: &str, generation: u64, phase: &'static str, tokens: usize) {
        self.update(|s| {
            s.current = Some(Current {
                request_id: request_id.to_owned(),
                config_generation: generation,
                phase,
                generated_tokens: tokens as u64,
            });
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
        spare: &mut crate::qwen4_exp::gpu::ExpertActivity,
    ) {
        let (entries, bytes, evictions) = cache.stats();

        gpu.copy_expert_activity(spare);

        let memory = MemoryStats {
            resources: gpu.memory_stats(),
            observed_at_uptime_seconds: self.started.elapsed().as_secs_f64(),
        };

        self.publish_observation(
            CacheStats {
                entries,
                bytes,
                evictions,
            },
            memory,
            spare,
        );
    }

    /// Swap complete snapshots while holding the lock; retain the old allocation
    /// as the worker's spare so the next observation can reuse it.
    pub(crate) fn publish_observation(
        &self,
        cache: CacheStats,
        memory: MemoryStats,
        spare: &mut crate::qwen4_exp::gpu::ExpertActivity,
    ) {
        self.update(|s| {
            s.cache = cache;
            s.memory = memory;

            std::mem::swap(&mut s.activity, spare);
        });
    }
}
