# Documentation

Run `mise run docs` to build the
[website](https://alfredr.github.io/cherenkov/) in `_site/`.
`.gitattributes` selects the files and `SUMMARY.md` sets the navigation.
Changes to `main` publish automatically.

- [Running](running.md): CLI, HTTP API, prefix caching and precision modes.
- [Server configuration](server-config.md): TOML, memory policy and control CLI.
- [Statistics](stats.md): summaries, detailed JSON, and the live dashboard.
- [Storage](storage.md): paths, downloads and packed expert stores.
- [Engine](engine.md): model execution, expert streaming and synchronization.
- [Developer options](developer-options.md): diagnostic environment variables.
- [Validation](validation.md): tests, linters and known limitations.
- [Serving architecture](serving-design.md): ownership, scheduling and cancellation.
- [Benchmarks](../benchmarks/README.md): suite, reports and pelicans.
- [Saved results](../results/README.md): measurements and generated answers.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/` | CLI, server, storage and shared runtime support |
| `src/server/` | HTTP framing, routes, request policy, scheduling, sessions and output |
| `src/control/` | Control state and statistics protocol |
| `src/cli_output/` | Terminal reports and dashboard |
| `src/runner/` | Resumable decoding and optional diagnostics |
| `src/qwen4_exp/` | Model config, packer, CPU reference and GPU execution |
| `kernels/` | Metal fragments grouped by common primitives and model subsystem |
| `tests/unit/` | Engine child-module tests, mirroring the source hierarchy |
| `xtask/src/`, `xtask/tests/` | Rust automation and its tests |
| `xtask/templates/` | HTML template used to generate benchmark galleries |
| `benchmarks/` | Workload definitions and measurement instructions |
| `results/` | Reviewed measurements; new local runs are ignored |

Model weights and packed stores use the configured [data directory](storage.md).
