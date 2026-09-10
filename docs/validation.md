# Validation

Current checks, September 10, 2026: Rust formatting and all 84 workspace tests
pass with the real model explicitly selected (64 engine, 20 automation).
The complexity limit of 12 passes. Strict Clippy still fails on nesting
findings at the experimental limit of two; Oxisym reports eight existing
engine findings and none in the Rust task runner. The AC benchmark
comparison below leaves a possible long-prompt decode regression unresolved.

Run correctness checks with no concurrent benchmark or inference process:

```sh
mise install
mise run fmt-check
mise run lint-md
mise run clippy
mise run oxisym
CHERENKOV_MODEL_DIR=/path/to/packed/model mise run test
mise exec -- cargo build --release --offline
```

Mise pins Rust 1.98.1 with Clippy and rustfmt, plus Node and markdownlint-cli2.
It uses rustup to install and select Rust; use `mise exec -- cargo ...` or
these tasks when the shell has not activated Mise. Oxisym additionally
requires `cargo-dylint` and `dylint-link`; Dylint selects the lint package's
own nightly toolchain. It is separate from the engine's stable compiler.

The 1.98.1 Clippy pass uses fixed-size array chunks in five places that
previously used `chunks_exact`. Chunk boundaries, remainder handling and
numerical iteration order are unchanged. Formatting and Markdown checks pass;
the full Clippy scan reports only the nesting findings described below.

`mise run fmt` applies Rust formatting. `mise run fix-md` applies available
Markdown fixes. Markdown linting covers all repository `.md` files except
ignored build and local-run directories. Prose wraps at 80 columns;
commands and table rows are exempt from line length, with compact tables.

## Repository automation

`cargo xtask` replaces the Python benchmark, reporting and smoke runners.
Its 20 tests cover timing separation, SVG validation, repetition detection,
scheduling, resume and cap increases, failed-sample exclusion, process
timeouts, interruption cleanup and README generation. All 32 baseline
median rows and the
80-job schedule match the previous runner. Existing saved measurements
and generated answers are retained. No new throughput suite was run for
this migration.

`mise run check-metal` compiles the same assembled sources as the engine.
All three libraries pass, with existing unused-constant and signedness
warnings.

## GPU and model checks

GPU tests require Metal. Model-dependent checks use `CHERENKOV_MODEL_DIR`,
or the default local cache when available. Without a packed model those
checks return early, so set the variable when validating model changes.
Low-bit GPU tests also require the corresponding cached `experts2.bin` and
`experts3.bin` files; they skip missing stores instead of building them.

The test-only child modules cover:

- QSA indexing, top-k ties, and selected attention against CPU/dense references.
- Greedy argmax ties, grid-strided input, and a partial final block.
- Real-weight prefill matrix projections against the CPU matvec.
- Q2/Q3 expert gate/up/down GEMMs against independently reconstructed
  original-code midpoints, including token-tile boundaries and output guards.
- Multi-chunk prefill through MTP verification and decode in Q4, Q3, Q2,
  and mixed Q4/Q2 modes; actual fetch bytes and resident precision are checked.
- Prefix snapshot restoration of every hybrid-state region, including
  recurrent/conv state, PLE histories, q8 KV scale planes, incomplete QSA
  blocks, and MTP enabled/disabled.
- Quantization, low-bit packing, CLI validation, context budgeting,
  prefix matching, and HTTP request validation.

Packing fixtures cover individual and combined expert targets, byte equality
with separate conversions, cached-store reuse, custom output directories,
relocated models, and failed-build manifest invalidation. The real cached
Q4/Q2/Q3 stores also pass `pack --experts 4,2,3` without changing file sizes
or modification times. No full-size conversion was needed for these checks.

The CPU `--check` path is a slow numerical reference. Its timings must
not enter throughput comparisons. Expert residency changes accumulation
order, so use fresh processes for output comparisons across revisions.
Prefix-state byte restoration does not guarantee identical generated
text across fresh and cached prefill boundaries.

## Server and control checks

```sh
# Each check starts and stops its own server:
cargo xtask smoke server --model /path/to/model
cargo xtask smoke control --model /path/to/model
```

The HTTP smoke client checks model discovery, JSON/SSE completions, usage,
prefix reuse, and errors. The control check covers configuration snapshots,
atomic reload rejection, output limits, prefix reuse and idle eviction.
Neither is a throughput benchmark. Reports go to ignored `results/*-smoke.json`.
The Rust control check passed after an initial intermittent socket error
(`Invalid argument`); its cause is not yet resolved.

The live M4 control check retained four-token limits for active and queued
requests across a reload, then used a two-token limit for the next request.
It reused 129 prompt tokens and reclaimed both checkpoints after idle expiry.
Observed Metal allocation was 20.44 GB; this was not a peak-memory measurement.

Unit tests also cover TOML and CLI precedence, explicit zero/false/adaptive
overrides, private sockets, bounded control frames, model-store policy,
allocation guards, and prompt token reuse at context/output boundaries.

## Storage and downloads

```sh
# Small network check; downloads metadata twice and verifies blob reuse:
cargo xtask smoke download
```

The pinned Hub and local configs declare `qwen4_exp` with a `qwen4_exp_text`
text model. The download check fetched six metadata files and reused the
unchanged blobs on its second run. It fetched no weight shards. Full model
download throughput and a fresh 104 GB base pack have not been tested.

Seven storage tests cover the portable root, pinned model identity, XDG
absolute overrides, empty/relative fallback, missing-home errors, path
validation and the disk-space guard. Explicit local model paths still work.

## Low-bit prefill checks

The [paired prefill suite](../results/prefill-lowbit-2026-09-09/README.md)
completed 36 samples. Default Q4 output matched byte for byte in all six
before/after pairs: 121, 521, and 3,221 prompt tokens, two rounds each.
The [mixed Q4/Q2 suite](../results/prefill-mixed-2026-09-09/README.md) adds
12 samples without a deadline cut.

Low-bit outputs intentionally change: the earlier implementation used Q4
experts during batched prefill even when decode used Q2/Q3. The GPU oracle
reconstructs low-bit weights from the original Q4 codes; CLI `--check` still
compares against the original Q4 model.

Use the [benchmark suite](../benchmarks/README.md) for complete-answer timing
and pelicans. Saved reports retain exact source/binary hashes, arguments,
power readings and excluded attempts.

## Readability checks

`Cargo.toml` enables Clippy's cognitive-complexity, excessive-nesting,
needless-else and redundant-else lints. `clippy.toml` now sets complexity to 12
and nesting to 2. These stricter limits are being evaluated; the tree does
not yet pass `cargo clippy --workspace --all-targets --offline -- -D warnings`.
See [Clippy's configuration reference](https://doc.rust-lang.org/stable/clippy/lint_configuration.html).

Prefer guard clauses (`ensure!`, `?`, `let ... else`, early returns and
`continue`) for errors, absent optional work and skipped loop entries.
Keep `if/else` expressions when they clearly select a value. Extract helpers
at meaningful operation boundaries, rather than splitting straight-line
Metal dispatch just to meet a line count. Clippy counts `impl` and function
bodies toward nesting: a single guard inside a method exceeds a limit of two.
The independent low-bit unpacking test has one explained `#[expect]` for its
word/lane/byte loops; production code has no exceptions to these limits.

The complexity pass separates config sections, prompt prefill paths, decode
lookahead/PLE setup, expert job preparation and reads, deadline handling, CPU
convolution and the three packed stores. The numerical loops retain their
iteration order; helpers borrow the existing buffers and add no GPU commands.
Config overrides retain explicit `if let` assignments.

All twelve previously flagged functions (nine production, three tests) now
pass the complexity limit of 12, with no complexity exemptions. The complete
Clippy scan still reports 387 nesting findings (356 production, 31 tests).
To check other lints while evaluating that separate nesting policy:

```sh
cargo clippy --workspace --all-targets --offline -- -D warnings -A clippy::excessive_nesting
```

All 53 Rust tests pass with the real model explicitly selected. Oxisym
retains eight existing findings. Before/after CPU `--check` runs produce
identical output and per-row numerical comparisons. Small packing fixtures
cover both n-gram shard naming conventions, multiple trunk layers and MTP:
all packed files, padding and manifests are byte-identical. No full checkpoint
repack was needed. The live server check passes for reload isolation, output
limits, prefix reuse and idle eviction.

The local paired comparison is saved in
`results/complexity-12-2026-09-09/report.json` with binary hashes, generated output,
per-process logs, token IDs, clock probes and start/end power readings.
Both binaries used Rust 1.94.0; these rates do not measure the 1.98.1 upgrade.
It uses fresh processes in ABBA order for 32-token samples and BAAB for
96-token samples, at Q4 with two adaptive drafts and context 512. EOS is
disabled; these are equal-token performance checks, not complete-answer
benchmarks. Model load and conversion are excluded from PP/TG rates.
All corresponding outputs and token IDs match.

| Workload | Tokens | Before decode tok/s | After decode tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Short prompt | 32 | 7.185 | 7.015 | -2.4% |
| Long prompt | 32 | 6.590 | 6.630 | +0.6% |
| Short prompt, longer continuation | 96 | 8.740 | 8.460 | -3.2% |

These are medians of two samples per binary per workload, on battery power
at each recorded start/end. The refactored 96-token runs ranged from 8.22 to
8.70 tok/s, showing appreciable variation. This small comparison does not
establish zero slowdown; it prompted the AC comparison below.

### AC follow-up, 2026-09-10

The same verified binaries were compared with two excluded warmups and four
measured samples per binary per prompt, in ABBA BAAB order. Every measured
sample generated 96 tokens with the same Q4/draft/context options. AC was
monitored every two seconds; all outputs and token IDs matched. Observed
Metal allocation remained 20.98 GB. Step counts and final-step dispatch
counts matched within each workload.

| Prompt | Before PP/s | After PP/s | Before TG/s | After TG/s | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short, 19 tokens | 4.40 | 4.40 | 8.425 | 8.470 | +0.5% |
| Long, 150 tokens | 14.05 | 14.10 | 8.400 | 7.855 | -6.5% |

Long-prompt decode was lower after the refactor in all four adjacent pairs:
-4.34%, -11.75%, -1.30%, and -0.23%. The changing gap and variable clock probes
make its magnitude uncertain, but the AC results do not clear the possible
regression. Prefill medians were nearly unchanged. No slower samples were
discarded. Full artifacts and the comparison script are saved locally under
`results/complexity-12-ac-2026-09-10/`; no source code changed during this retest.
