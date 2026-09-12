//! Check that a PR author's GitHub login appears in AUTHORS.

use anyhow::{Context, Result, ensure};
use regex::Regex;
use std::{fs, path::Path};

pub fn check(path: &Path, login: &str) -> Result<()> {
    let authors = fs::read_to_string(path).with_context(|| {
        format!(
            "reading {}; add your contribution acknowledgment there",
            path.display()
        )
    })?;
    let entry = Regex::new(r"^.+ <[^<>\s]+@[^<>\s]+> \(@([^()\s]+)\)$")?;
    let listed = authors
        .lines()
        .filter_map(|line| entry.captures(line.trim()))
        .any(|fields| fields[1].eq_ignore_ascii_case(login));

    ensure!(
        listed,
        "Read the contribution terms in AUTHORS, then acknowledge them by adding \
         your own entry: Your Name <your Git email> (@{login}). \
         You may use your GitHub noreply email."
    );

    Ok(())
}
