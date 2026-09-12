//! Bounded read counters. Reader threads publish once per completed record.

use serde::{Deserialize, Serialize};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct ReadStats {
    pub requested_reads: u64,
    pub requested_bytes: u64,
    pub completed_reads: u64,
    pub failed_reads: u64,
    pub completed_bytes: u64,
    pub read_seconds: f64,
    pub max_read_seconds: f64,
}

impl ReadStats {
    pub fn add(&mut self, other: Self) {
        self.requested_reads += other.requested_reads;
        self.requested_bytes += other.requested_bytes;
        self.completed_reads += other.completed_reads;
        self.failed_reads += other.failed_reads;
        self.completed_bytes += other.completed_bytes;
        self.read_seconds += other.read_seconds;
        self.max_read_seconds = self.max_read_seconds.max(other.max_read_seconds);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ReadSource {
    Prefill,
    Demand,
    Prefetch,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct ReadSources {
    pub prefill: ReadStats,
    pub demand: ReadStats,
    pub prefetch: ReadStats,
}

impl ReadSources {
    fn source(&mut self, source: ReadSource) -> &mut ReadStats {
        match source {
            ReadSource::Prefill => &mut self.prefill,
            ReadSource::Demand => &mut self.demand,
            ReadSource::Prefetch => &mut self.prefetch,
        }
    }

    pub fn add(&mut self, other: Self) {
        self.prefill.add(other.prefill);
        self.demand.add(other.demand);
        self.prefetch.add(other.prefetch);
    }

    pub fn total(self) -> ReadStats {
        let mut total = self.prefill;

        total.add(self.demand);
        total.add(self.prefetch);

        total
    }
}

#[derive(Debug, Clone)]
pub struct ReadTracker(Arc<Mutex<Vec<[ReadSources; 3]>>>);

impl ReadTracker {
    pub fn new(layers: usize) -> Self {
        Self(Arc::new(Mutex::new(vec![
            [ReadSources::default(); 3];
            layers
        ])))
    }

    pub fn snapshot(&self) -> Vec<[ReadSources; 3]> {
        self.0.lock().unwrap().clone()
    }

    pub fn ticket(&self, layer: usize, kind: u8, source: ReadSource, bytes: usize) -> ReadTicket {
        let mut layers = self.0.lock().unwrap();
        let stats = layers[layer][kind as usize].source(source);
        stats.requested_reads += 1;
        stats.requested_bytes += bytes as u64;

        ReadTicket {
            tracker: self.clone(),
            layer,
            kind,
            source,
            bytes,
            finished: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReadTicket {
    tracker: ReadTracker,
    layer: usize,
    kind: u8,
    source: ReadSource,
    bytes: usize,
    finished: Arc<AtomicBool>,
}

impl ReadTicket {
    pub fn done(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    pub fn measure(&self, read: impl FnOnce() -> usize) {
        let started = Instant::now();
        let bytes = read();

        self.finish(bytes, started.elapsed().as_secs_f64());
    }

    fn finish(&self, bytes: usize, seconds: f64) {
        let mut layers = self.tracker.0.lock().unwrap();
        let stats = layers[self.layer][self.kind as usize].source(self.source);
        stats.completed_reads += u64::from(bytes == self.bytes);
        stats.failed_reads += u64::from(bytes != self.bytes);
        stats.completed_bytes += bytes as u64;
        stats.read_seconds += seconds;
        stats.max_read_seconds = stats.max_read_seconds.max(seconds);

        self.finished.store(true, Ordering::Release);
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/qwen4_exp/gpu/read_stats.rs"]
mod tests;
