# Validation

Run correctness checks without concurrent inference or benchmarks:

```sh
mise install
mise run fmt-check
mise run lint-md
CHERENKOV_MODEL_DIR=/path/to/packed/model mise run test
mise run check-metal
mise run clippy
mise run oxisym
```

`mise.toml` pins Rust, rustfmt, Clippy, styling-lint, Node and markdownlint-cli2.
Oxisym additionally requires `cargo-dylint` and `dylint-link` and selects its own
nightly toolchain. `mise run fmt` formats Rust; `mise run fix-md` fixes
Markdown. Markdown checks cover every `.md` file outside build directories.

`mise run lint-spacing` checks statement spacing across Rust source and tests.
`mise run fix-spacing` runs rustfmt, then adds the missing blank lines.
[styling-lint](https://github.com/visionsofparadise/styling-lint) keeps consecutive
assignments and calls together and separates other statement groups. These are
syntax rules, so review the diff for places where related steps belong together.
The existing code still has spacing findings; this check is separate from the
correctness checks above.

## Tests

Engine unit tests live in `tests/unit/`, mirroring their source modules.
They are compiled as child modules so they can test private implementation
details. Rust task-runner tests live in `xtask/tests/`.

Prompt tests compare the unmodified checkpoint template against 20 independently
generated Transformers/Jinja2 reference contexts. Rendered bytes and template
errors are checked without a model. With `CHERENKOV_MODEL_DIR`, token IDs are
also checked using the checkpoint tokenizer and verified input hashes. These
tests need no Python runtime; [fixture provenance](../tests/fixtures/prompt/README.md)
records the reference versions and regeneration procedure.

CPU tests cover configuration, CLI validation, packing, quantization, context
budgets, prefix matching, storage paths and HTTP request handling. Sampling
checks cover distributions, seeds, pending-token suspension, RNG continuation,
cancellation/publication ordering, session pinning, eviction and memory failure.
Packing
fixtures exercise individual and combined targets, cache reuse, custom output
directories and failed-build manifest invalidation.

GPU tests require Apple Silicon and Metal. Real-weight tests also require
`CHERENKOV_MODEL_DIR`, or a model in the default managed location. They skip
when the model is absent; low-bit checks also skip missing Q2/Q3 stores.
They verify attention, argmax, expert projections, prefill and complete
hybrid-state checkpoint restoration against CPU or independent references.
`check-metal` compiles the same assembled shader libraries as the engine and
fails when `kernels/.clangd` is stale; `mise run clangd` regenerates it.

Task-runner tests cover timing separation, SVG validation, cycle detection,
resume rules, process cleanup, report generation and README updates.

## Live checks

```sh
cargo xtask smoke server --model /path/to/model
cargo xtask smoke control --model /path/to/model
cargo xtask smoke sessions --model /path/to/model
cargo xtask smoke download
```

Server checks cover JSON/SSE completions, usage, prefix reuse and errors.
Control checks cover configuration reloads, limits and cache expiry. Each
starts and stops its own server. Session checks exercise concurrent progress,
queued/prefill/decode cancellation, rollback and continued turns. They also
cover four mixed greedy/sampled sessions, disconnects, memory admission,
request-capacity rejection and ID reuse, and recovery after response-limit
failures. Unit tests exercise full writer queues and cancellation/publication
races without timing-sensitive sockets. The download
check fetches only metadata
and verifies cache reuse. Reports go to ignored `results/*-smoke.json`.

## Readability checks

Clippy checks cognitive complexity at 12, nesting at 2, and unnecessary
`else` branches. Prefer guard clauses and helpers at meaningful operation
boundaries. Keep numerical iteration order and GPU dispatch order explicit;
do not split them merely to meet a line count.

The complexity limit passes. The experimental nesting limit still flags
existing code, including simple guards inside methods. To check other lints:

```sh
cargo clippy --workspace --all-targets --offline -- -D warnings -A clippy::excessive_nesting
```

Oxisym currently reports eight existing structural-similarity findings.

## Performance and known limitations

Use the [benchmark suite](../benchmarks/README.md) for full answers, phase
timings and pelicans. [Saved results](../results/README.md) identify their
measured revisions; correctness tests do not establish current throughput.

The CPU `--check` oracle is deliberately slow. Fresh and cached runs can
produce different close argmax decisions because expert accumulation order
changes. Byte-exact state restoration does not imply identical generated text.

A prior local AC comparison found a possible 6.5% long-prompt decode
regression after readability changes; it remains unresolved. Passing tests
and the published baseline do not rule it out. Full model download throughput
and a fresh full-size pack have not been validated.
