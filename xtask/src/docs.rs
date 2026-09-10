//! Stage the repository's Markdown and public assets for mdBook.

use crate::util;
use anyhow::{Context, Result, ensure};
use std::{fs, path::Path, process::Command};

pub fn build() -> Result<()> {
    let root = util::root();
    let staging = tempfile::tempdir()?;

    stage(&root, staging.path())?;

    let status = Command::new("mdbook")
        .arg("build")
        .arg(staging.path())
        .arg("--dest-dir")
        .arg(root.join("_site"))
        .status()
        .context("running mdbook; install it with mise install github:rust-lang/mdBook")?;

    ensure!(status.success(), "mdBook build failed");

    Ok(())
}

pub fn stage(root: &Path, destination: &Path) -> Result<()> {
    let tracked = Command::new("git")
        .args(["ls-files", "-z", "--", ":(attr:site)"])
        .current_dir(root)
        .output()?;
    let paths = util::checked(tracked)?;

    fs::create_dir_all(destination)?;
    fs::copy(root.join("book.toml"), destination.join("book.toml"))?;

    // Use working-tree contents, but only tracked files: local benchmark runs
    // and model downloads must never become website assets.
    for name in paths.split('\0').filter(|name| !name.is_empty()) {
        let path = Path::new(name);
        let target = destination.join(path);

        fs::create_dir_all(target.parent().context("site file parent")?)?;
        fs::copy(root.join(path), &target)?;
    }

    Ok(())
}
