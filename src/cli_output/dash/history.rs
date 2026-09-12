//! Bounded interval history. Missing observations never become zero activity.

use cherenkov::control::stats::{Snapshot, Summary};
use std::collections::VecDeque;

const HISTORY_SECONDS: f64 = 120.0;
const MAX_SAMPLES: usize = 120;
pub(super) const BYTES_PER_MB: f64 = 1_000_000.0;

#[derive(Clone, Copy)]
struct Counters {
    time: f64,
    elapsed: f64,
    bytes: u64,
    wait: f64,
    stage: f64,
    gap: f64,
    gpu: bool,
}

impl From<&Snapshot<Summary>> for Counters {
    fn from(snapshot: &Snapshot<Summary>) -> Self {
        let phases = snapshot.data.streaming.phases;

        Self {
            time: snapshot.observation.observed_at_uptime_seconds,
            elapsed: snapshot.observation.elapsed_seconds,
            bytes: snapshot.data.reads.completed_bytes,
            wait: phases.prefetch_wait_seconds + phases.demand_wait_seconds,
            stage: phases.resident_seconds + phases.fetched_stage_seconds,
            gap: phases.router_to_resident_seconds + phases.resident_to_fetched_seconds,
            gpu: snapshot.observation.gpu_timestamps_available,
        }
    }
}

#[derive(Default)]
pub(super) struct History {
    pub samples: VecDeque<Sample>,
    previous: Option<Counters>,
    disconnected: bool,
}

pub(super) struct Sample {
    pub time: f64,
    // MB/s, CPU read wait ms/s, measured GPU stage coverage percent.
    pub values: [Option<f64>; 3],
}

impl History {
    pub fn disconnect(&mut self) {
        self.disconnected = true;
    }

    pub fn observe(&mut self, snapshot: &Snapshot<Summary>) {
        let current = Counters::from(snapshot);

        if ![
            current.time,
            current.elapsed,
            current.wait,
            current.stage,
            current.gap,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
        {
            self.disconnect();

            return;
        }

        let Some(previous) = self.previous else {
            self.previous = Some(current);

            self.push(current.time, [None; 3]);

            return;
        };

        if current.reset_since(previous) {
            self.samples.clear();

            self.previous = Some(current);
            self.disconnected = false;

            self.push(current.time, [None; 3]);

            return;
        }

        // The server can return the same worker observation more than once.
        if current.time == previous.time {
            return;
        }

        let values = if self.disconnected {
            [None; 3]
        } else {
            current.rates_since(previous)
        };
        self.previous = Some(current);
        self.disconnected = false;

        self.push(current.time, values);
    }

    fn push(&mut self, time: f64, values: [Option<f64>; 3]) {
        self.samples.push_back(Sample { time, values });

        while self.samples.len() > MAX_SAMPLES
            || self
                .samples
                .front()
                .is_some_and(|sample| time - sample.time > HISTORY_SECONDS)
        {
            self.samples.pop_front();
        }
    }
}

impl Counters {
    fn reset_since(self, previous: Self) -> bool {
        self.time < previous.time
            || self.elapsed < previous.elapsed
            || self.bytes < previous.bytes
            || self.wait < previous.wait
            || self.stage < previous.stage
            || self.gap < previous.gap
            // Uptime minus model age identifies a new load in the same server.
            || ((self.time - self.elapsed) - (previous.time - previous.elapsed)).abs() > 0.1
    }

    fn rates_since(self, previous: Self) -> [Option<f64>; 3] {
        let seconds = self.time - previous.time;
        let stage = self.stage - previous.stage;
        let gap = self.gap - previous.gap;
        let coverage =
            (self.gpu && previous.gpu && stage + gap > 0.0).then(|| 100.0 * stage / (stage + gap));

        [
            Some((self.bytes - previous.bytes) as f64 / seconds / BYTES_PER_MB),
            Some((self.wait - previous.wait) * 1000.0 / seconds),
            coverage,
        ]
    }
}
