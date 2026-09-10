//! Request-owned sampling state. GPU logits are borrowed after synchronization.

use anyhow::{Result, ensure};
use clap::Args;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Args, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    /// Sampling temperature; 0 keeps greedy decoding
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,
    /// Nucleus probability, in (0, 1]
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f64,
    /// Keep the highest K logits; 0 keeps the whole vocabulary
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,
    /// Seed for this request's random stream
    #[arg(long)]
    pub seed: Option<u64>,
    /// Penalty applied once to tokens present in the prompt or output
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub presence_penalty: f64,
    /// Penalty multiplied by each token's occurrence count
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub frequency_penalty: f64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl Sampling {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && (0.0..=2.0).contains(&self.temperature),
            "temperature must be finite and in 0..2"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be finite and in (0, 1]"
        );
        ensure!(self.top_k <= 1_000_000, "top_k must be 0..1000000");

        for (name, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            ensure!(
                value.is_finite() && (-2.0..=2.0).contains(&value),
                "{name} must be finite and in -2..2"
            );
        }

        Ok(())
    }

    /// Only the unmodified greedy distribution can use argmax MTP verification.
    pub fn greedy(&self) -> bool {
        self.temperature == 0.0 && self.presence_penalty == 0.0 && self.frequency_penalty == 0.0
    }

    pub(crate) fn from_request(value: &serde_json::Value, defaults: &Self) -> Result<Self> {
        let mut fields = serde_json::to_value(defaults)?;

        for key in [
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ] {
            if let Some(v) = value.get(key).filter(|v| !v.is_null()) {
                fields[key] = v.clone();
            }
        }

        let result: Self = serde_json::from_value(fields)?;

        result.validate()?;

        Ok(result)
    }
}

pub(crate) struct Sampler {
    options: Sampling,
    rng: StdRng,
    counts: Vec<u32>,
    candidates: Vec<(u32, f64)>,
}

impl Sampler {
    pub(crate) fn set_rng(&mut self, rng: StdRng) {
        self.rng = rng;
    }

    pub(crate) fn rng(&self) -> StdRng {
        self.rng.clone()
    }

    pub(crate) fn new(options: &Sampling, prompt: &[u32], vocab: usize) -> Result<Self> {
        options.validate()?;

        let mut sampler = Self {
            options: options.clone(),
            rng: options
                .seed
                .map(StdRng::seed_from_u64)
                .unwrap_or_else(StdRng::from_os_rng),
            counts: vec![0; vocab],
            candidates: Vec::with_capacity(vocab),
        };

        for &token in prompt {
            sampler.accept(token)?;
        }

        Ok(sampler)
    }

    pub(crate) fn accept(&mut self, token: u32) -> Result<()> {
        let count = self
            .counts
            .get_mut(token as usize)
            .ok_or_else(|| anyhow::anyhow!("token exceeds vocabulary"))?;
        *count = count.saturating_add(1);

        Ok(())
    }

    pub(crate) fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        ensure!(
            logits.len() == self.counts.len(),
            "logit vocabulary mismatch"
        );
        self.candidates.clear();

        for (id, (&logit, &count)) in logits.iter().zip(&self.counts).enumerate() {
            ensure!(
                !logit.is_nan() && logit != f32::INFINITY,
                "non-finite model logits"
            );

            let penalty = self.options.frequency_penalty * f64::from(count)
                + if count > 0 {
                    self.options.presence_penalty
                } else {
                    0.0
                };

            self.candidates
                .push((id as u32, f64::from(logit) - penalty));
        }

        let best = self
            .candidates
            .iter()
            .min_by(|a, b| rank(a, b))
            .copied()
            .ok_or_else(|| anyhow::anyhow!("empty vocabulary"))?;

        ensure!(best.1.is_finite(), "no finite candidate logits");

        if self.options.temperature == 0.0 {
            return Ok(best.0);
        }

        let k = self.options.top_k;

        if k > 0 && k < self.candidates.len() {
            self.candidates.select_nth_unstable_by(k, rank);
            self.candidates.truncate(k);
        }

        if self.options.top_p < 1.0 {
            self.candidates.sort_unstable_by(rank);
        }

        let mut total = 0.0;

        for (_, weight) in &mut self.candidates {
            *weight = ((*weight - best.1) / self.options.temperature).exp();
            total += *weight;
        }

        let (keep, mass) = nucleus(&self.candidates, total * self.options.top_p);
        let draw = self.rng.random::<f64>() * mass;
        let mut cumulative = 0.0;

        for &(token, weight) in &self.candidates[..keep] {
            cumulative += weight;

            if draw < cumulative {
                return Ok(token);
            }
        }

        Ok(self.candidates[keep - 1].0)
    }
}

fn rank(a: &(u32, f64), b: &(u32, f64)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

fn nucleus(candidates: &[(u32, f64)], threshold: f64) -> (usize, f64) {
    let mut mass = 0.0;

    for (i, &(_, weight)) in candidates.iter().enumerate() {
        mass += weight;

        if mass >= threshold {
            return (i + 1, mass);
        }
    }

    (candidates.len(), mass)
}

#[cfg(test)]
#[path = "../tests/unit/sampling.rs"]
mod tests;
