use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}

pub fn absolute(path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}

pub fn command(args: &[&str]) -> Command {
    let mut command = Command::new(args[0]);
    command.args(&args[1..]).current_dir(root());
    command
}

pub fn checked(output: Output) -> Result<String> {
    ensure!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn output(args: &[&str]) -> Result<String> {
    checked(
        command(args)
            .output()
            .with_context(|| format!("running {}", args[0]))?,
    )
}

pub fn digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

pub fn files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            result.extend(files(&entry.path())?);
        } else {
            result.push(entry.path());
        }
    }
    result.sort();
    Ok(result)
}

pub fn source_digest() -> Result<String> {
    let root = root();
    let mut sources = files(&root.join("src"))?;
    sources.extend(files(&root.join("kernels"))?);
    sources.sort();
    sources.extend([root.join("Cargo.toml"), root.join("Cargo.lock")]);
    let mut hash = Sha256::new();
    for path in sources {
        hash.update(path.strip_prefix(&root)?.to_string_lossy().as_bytes());
        hash.update(fs::read(path)?);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

pub fn json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path)?).with_context(|| path.display().to_string())
}

pub fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().context("output parent")?)?;
    use std::io::Write;
    writeln!(file, "{}", serde_json::to_string_pretty(value)?)?;
    file.persist(path)?;
    Ok(())
}

pub fn utc() -> Result<String> {
    output(&["date", "-u", "+%Y%m%dT%H%M%SZ"])
}

pub fn build() -> Result<()> {
    ensure!(
        command(&[
            "cargo",
            "build",
            "--release",
            "--offline",
            "-p",
            "cherenkov"
        ])
        .status()?
        .success(),
        "release build failed"
    );
    Ok(())
}
