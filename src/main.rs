use anyhow::{Context, Result};
use cherenkov::{
    config::{self, Overrides, Source},
    control, download,
    options::Options,
    qwen4_exp, runner,
    storage::{DEFAULT_REPO, DEFAULT_REVISION, Paths},
};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "cherenkov",
    version,
    about = "qwen4-exp text inference on Apple Silicon",
    next_line_help = false,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// Put model data, scratch and config under one root instead of XDG directories
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[arg(required = true)]
    model_dir: Option<PathBuf>,
    #[arg(required = true)]
    prompt: Option<String>,
    #[command(flatten)]
    options: Options,
}

#[derive(Args)]
#[command(next_line_help = false)]
struct Serve {
    /// Server configuration file (TOML)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Print resolved TOML without loading the model
    #[arg(long)]
    print_config: bool,
    #[command(flatten)]
    overrides: Overrides,
}

#[derive(Subcommand)]
enum Command {
    /// Serve OpenAI-compatible completions and local control commands
    Serve(Box<Serve>),
    /// Query the resident server
    Status {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Inspect or reload the resident server's configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
        #[arg(long, global = true)]
        socket: Option<PathBuf>,
    },
    /// Show resolved data, scratch, config and default model locations
    Paths,
    /// Download the supported checkpoint, with native Hugging Face/Xet transfers
    Download {
        #[arg(default_value = DEFAULT_REPO)]
        repo: String,
        /// Branch, tag or commit; defaults to the measured checkpoint revision
        #[arg(long, default_value = DEFAULT_REVISION)]
        revision: String,
        /// Hugging Face access token (alternatively HF_TOKEN or an existing HF login)
        #[arg(long)]
        hf_token: Option<String>,
        /// Fetch configuration and tokenizer without the weight shards
        #[arg(long)]
        metadata_only: bool,
    },
    /// Prepare the aligned 4-bit store and selected expert variants
    Pack {
        model_dir: Option<PathBuf>,
        /// Store directory (defaults to MODEL_DIR/packed, or MODEL_DIR if already packed)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Expert precisions (4,3,2); comma-separated or repeated; reuse existing stores
        #[arg(long, default_value = "4", value_delimiter = ',', num_args = 1..,
            value_parser = clap::value_parser!(u32).range(2..=4))]
        experts: Vec<u32>,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    Show,
    Reload,
}

impl Serve {
    fn source(mut self, root: Option<PathBuf>) -> Result<Source> {
        self.overrides.root = root.as_deref().map(config::absolute).transpose()?;
        let path = match self.config {
            Some(path) => Some(config::absolute(&path)?),
            None => {
                let path = Paths::new(self.overrides.root.as_deref())?.config;

                match std::fs::metadata(&path) {
                    Ok(_) => Some(path),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                    Err(e) => return Err(e.into()),
                }
            }
        };
        self.overrides.model_dir = self
            .overrides
            .model_dir
            .as_deref()
            .map(config::absolute)
            .transpose()?;
        self.overrides.socket = self
            .overrides
            .socket
            .as_deref()
            .map(config::absolute)
            .transpose()?;

        Ok(Source {
            path,
            overrides: self.overrides,
        })
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Serve(args)) => {
            let print = args.print_config;
            let source = args.source(cli.root)?;

            if print {
                println!("{}", toml::to_string_pretty(&source.resolve()?)?);

                Ok(())
            } else {
                cherenkov::server::serve(source)
            }
        }
        Some(Command::Status { socket, json }) => {
            let result = control::query(
                &socket.unwrap_or_else(config::default_socket),
                control::Command::Status,
            )?;

            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                let s = &result["stats"];

                println!(
                    "{} | {} active, {} queued | {} completed, {} failed\n{} tokens | cache {} entries, {} bytes\nMetal {} bytes (last observation)",
                    if s["ready"] == true {
                        "ready"
                    } else {
                        "loading"
                    },
                    s["active_requests"],
                    s["queued_requests"],
                    s["completed_requests"],
                    s["failed_requests"],
                    s["generated_tokens"],
                    s["cache"]["entries"],
                    s["cache"]["bytes"],
                    s["memory"]["metal_allocated_bytes_observed"]
                );
            }

            Ok(())
        }
        Some(Command::Config { action, socket }) => {
            let command = match action {
                ConfigAction::Show => control::Command::ConfigShow,
                ConfigAction::Reload => control::Command::ConfigReload,
            };
            let result = control::query(&socket.unwrap_or_else(config::default_socket), command)?;

            println!("{}", serde_json::to_string_pretty(&result)?);

            Ok(())
        }
        Some(Command::Paths) => {
            let paths = Paths::new(cli.root.as_deref())?;

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "data": paths.data, "scratch": paths.scratch, "config": paths.config,
                    "default_model": paths.default_model(),
                }))?
            );

            Ok(())
        }
        Some(Command::Download {
            repo,
            revision,
            hf_token,
            metadata_only,
        }) => {
            let paths = Paths::new(cli.root.as_deref())?;

            // SAFETY: download is a standalone command. No worker or transfer
            // threads have started; Xet reads this setting when its session starts.
            if std::env::var_os("HF_XET_CACHE").is_none() {
                unsafe {
                    std::env::set_var("HF_XET_CACHE", paths.scratch.join("xet"));
                }
            }

            let model = download::run(
                &paths,
                download::Download {
                    repo: &repo,
                    revision: &revision,
                    token: hf_token.as_deref(),
                    metadata_only,
                },
            )?;

            println!("{}", model.display());

            Ok(())
        }
        Some(Command::Pack {
            model_dir,
            output,
            experts,
        }) => {
            let model_dir = match model_dir {
                Some(dir) => dir,
                None => Paths::new(cli.root.as_deref())?.default_model(),
            };

            qwen4_exp::Qwen4ExpConfig::load(&model_dir)?;

            qwen4_exp::pack::prepare(&model_dir, output.as_deref(), &experts)
        }
        None => runner::run(
            &cli.model_dir.context("model directory required")?,
            &cli.prompt.context("prompt required")?,
            &cli.options,
        ),
    }
}

#[cfg(test)]
#[path = "../tests/unit/cli.rs"]
mod tests;
