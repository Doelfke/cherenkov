# Native Cherenkov benchmark

Requires Rust, Metal, and a packed qwen4-exp checkpoint. Run on AC power
with other GPU work stopped. The runner lives in `xtask/`; use
`mise exec -- cargo ...` if Mise is not activated in your shell.

```sh
cargo xtask bench /path/to/model --dry-run
cargo xtask bench /path/to/model --build-stores
# A shorter comparison:
cargo xtask bench /path/to/model --cases code,prose --rounds 1
# Save a named run, or resume it after reconnecting power:
cargo xtask bench /path/to/model --output results/comparison
cargo xtask bench /path/to/model --output results/comparison --resume
cargo test -p xtask
```

The runner builds the release binary offline unless `--binary` is supplied.
`--build-stores` permits first-use low-bit construction: about 39 GB for
2-bit and 54 GB for 3-bit, in addition to the packed source. It retains
existing stores. Construction belongs to load time, never prefill time.
Without this option, missing stores are rejected before any inference.

## Workloads and measurement

`suite.json` contains all prompts and options. The default is **80 fresh
processes**: six completion workloads (linked-list code, LRU cache code,
binary-search debugging, a hash-table explanation, scheduling arithmetic,
and structured JSON extraction), three rounds at each of four settings;
plus one long-prefill case and one pelican per setting.

Every case generates a complete answer through EOS. Safety caps are
4,096 tokens by default, 7,168 for LRU code, reasoning, and pelicans,
and 1,024 for the document summary; hitting a cap marks
the sample incomplete and keeps it out of completion medians. The suite
continues so one unsuccessful answer does not hide the other workloads.
There is no forced fixed-length continuation after EOS. Configurations rotate within
interleaved rounds, and drawings run after the timed workload phase.
The long repeated-document prompt's token count is measured, not assumed.
Context capacity is 8,192 throughout; this affects expert-pool capacity.
Answers can differ in content and length, so this is an end-to-end
completion workload comparison, not equal-work kernel throughput.
Completion means reaching EOS; the suite does not grade the generated
code or factual answers for correctness.

The four settings are: 4-bit, 4-bit
resident/2-bit misses **with cut 0.08**, 3-bit, and 2-bit. The mixed
setting changes both precision and deadline behavior; its output is
not reproducible. Its gains must not be attributed to precision alone.
All use two adaptive MTP drafts and the adaptive pool. Routed-expert
precision applies to both prefill and decode. Uniform low-bit runs use
their compressed store in both phases; mixed prefill keeps Q4 pool entries
and streams transient misses at the selected miss precision. The saved
September 9 baseline predates this change and used Q4 batched prefill.

Each sample starts a new model process. The OS file cache is retained;
this is not a cold-SSD benchmark. No `--check`, warm `--repeat`, server
prefix cache, or developer environment overrides are active. Power is
sampled before/after each process and every 30 seconds; shorter power
transitions can go undetected. A power-source change invalidates the sample.
Reported Metal allocations above 25 decimal GB also invalidate it; these
readings are taken after prefill scratch is released, so they do not
measure its transient peak or cap process memory/the OS file cache.

The `gpu_active_ms` field comes from the command buffer's GPU end timestamp
minus its start timestamp. It is not a utilization counter. `io_wait_ms`
times the host's expert-read servicing interval, which overlaps GPU work;
do not add the two as independent components of decode time.

## Output

- `report.json`: every sample, exact arguments, load/PP/TG timings, steps,
  MTP acceptance, GPU/IO time, memory, clock probes, power samples, source
  commit/hash, binary hash, and model metadata hashes.
- `summary.md`: per-case/configuration medians, including output lengths
  and completion times, excluding invalid runs.
- `outputs/`: full generated text, including reasoning/code fences.
- `pelicans/`: complete XML-validated SVGs extracted without editing.
- `gallery.html`: browser-ready speed comparisons, expandable phase timings,
  and passive images with generation rates and links to full output. It
  opens directly from disk; no server or Markdown renderer is needed.

Drawings are a visual artifact, not an accuracy metric. Element count
and the requested 120-element budget are recorded. A missing/truncated
or malformed SVG is recorded as an invalid artifact, with its output retained.
SVG tg/s is listed separately because generated lengths/content differ.

The report is flushed after every sample. `--resume` skips successful
and content-invalid samples only when binary, suite, model metadata, and
selections match. Engine errors, memory-limit failures, and power changes
stop the suite and are retried on resume.
Failed samples retain a diagnostic tail; routine engine logs are temporary.
All outputs live under `results/`, with a UTC timestamp by default.
Use `--output results/<name>` to name a run. New results are ignored by Git;
use `git add -f results/<name>` to retain a reviewed run with the source.

Regenerate summaries and the gallery from an existing report without
loading a model: `cargo xtask summarize results/<name>`.

`--update-readme` refreshes the root README after a completed benchmark.
Use `cargo xtask readme results/<name>` to regenerate it from an existing
`report.json`. The generated section contains measured tables and links
to the saved pelican SVGs; the surrounding prose stays hand-written.

To retry a capped workload with more room while keeping completed EOS
samples, use `--case-cap code-lru=7168 --resume` with the same output path.
All other settings must match. Prior capped outputs move to `attempts/`,
and the old signature is retained in `suite_revisions`. A lower cap or a
changed prompt/context/model is not accepted as the same comparison.

Every 30 seconds the runner checks for four exact repetitions of a
32–512-word block at the output tail, retaining numbers so a progressing
sequence is not treated as a cycle. A detected cycle stops that sample,
retains the text and evidence, and continues the suite. This conservative
heuristic does not prove the absence of all repetition or semantic loops.

## Paired prefill comparison

`cargo xtask prefill` measures three prompt lengths using fresh processes, uniform
4/3/2-bit experts, and alternating before/after order. It emits exactly
eight decode tokens to check the handoff; these are throughput samples,
not complete-answer or decode-speed comparisons. Add `--configs 4/2` for
4-bit resident experts with 2-bit misses, or `--configs 4/3` for 3-bit misses;
these measurements use no deadline cut. Loading and conversion
are separate; an unexpected store build invalidates the sample.

```sh
cargo xtask prefill --before /path/to/old/cherenkov \
  --after target/release/cherenkov --model /path/to/model \
  --out results/prefill-comparison --rounds 2
```

The report preserves binary hashes, the source diff, prompts, raw telemetry,
generated text, and power readings. Keep both cached low-bit stores ready
before starting. Power is checked before/after each process and every
30 seconds; shorter transitions can be missed.
Run with no concurrent inference or GPU tests.
