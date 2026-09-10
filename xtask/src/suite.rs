use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Configuration {
    pub id: String,
    pub label: String,
    pub args: Vec<String>,
    pub store_bits: Option<u8>,
    pub reproducible_cut: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    pub max_tokens: usize,
    pub stop: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ctx: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounds: Option<usize>,
}

impl Case {
    pub fn prompt(&self) -> String {
        if let Some(prompt) = &self.prompt {
            return prompt.clone();
        }
        self.repeat_text
            .as_deref()
            .unwrap_or("")
            .repeat(self.repeat.unwrap_or(0))
            + self.suffix.as_deref().unwrap_or("")
    }
}

#[derive(Deserialize)]
pub struct Suite {
    pub rounds: usize,
    pub max_ctx: usize,
    pub configurations: Vec<Configuration>,
    pub cases: Vec<Case>,
}

pub fn select<T: Clone>(
    items: &[T],
    selection: Option<&str>,
    id: impl Fn(&T) -> &str,
) -> Result<Vec<T>> {
    let Some(selection) = selection else {
        return Ok(items.to_vec());
    };
    let mut seen = HashSet::new();
    selection
        .split(',')
        .map(|name| {
            ensure!(seen.insert(name), "repeated selection: {name}");
            items
                .iter()
                .find(|v| id(v) == name)
                .cloned()
                .with_context(|| format!("unknown selection: {name}"))
        })
        .collect()
}

pub fn schedule(
    configs: &[Configuration],
    cases: &[Case],
    rounds: usize,
) -> Vec<(usize, usize, usize)> {
    let mut jobs = Vec::new();
    for round in 0..rounds {
        for (case_index, case) in cases.iter().enumerate() {
            if case.kind == "svg" || round >= case.rounds.unwrap_or(rounds) {
                continue;
            }
            for index in 0..configs.len() {
                jobs.push((round, (index + round) % configs.len(), case_index));
            }
        }
    }
    for (index, case) in cases.iter().enumerate() {
        if case.kind != "svg" {
            continue;
        }
        jobs.extend((0..configs.len()).map(|c| (0, c, index)));
    }
    jobs
}

pub fn only_higher_caps(old: &Value, new: &Value) -> bool {
    let (mut old_other, mut new_other) = (old.clone(), new.clone());
    let (Some(a), Some(b)) = (old_other.as_object_mut(), new_other.as_object_mut()) else {
        return false;
    };
    let (Some(a), Some(b)) = (a.remove("cases"), b.remove("cases")) else {
        return false;
    };
    if old_other != new_other {
        return false;
    }
    let (Some(a), Some(b)) = (a.as_array(), b.as_array()) else {
        return false;
    };
    if a.len() != b.len() {
        return false;
    }
    let mut increased = false;
    for (a, b) in a.iter().zip(b) {
        let (mut a, mut b) = (a.clone(), b.clone());
        let (Some(a), Some(b)) = (a.as_object_mut(), b.as_object_mut()) else {
            return false;
        };
        let (Some(x), Some(y)) = (
            a.remove("max_tokens").and_then(|v| v.as_u64()),
            b.remove("max_tokens").and_then(|v| v.as_u64()),
        ) else {
            return false;
        };
        if a != b || y < x {
            return false;
        }
        increased |= y > x;
    }
    increased
}
