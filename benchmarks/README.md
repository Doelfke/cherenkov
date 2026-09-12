# Benchmarks

The suite requires Rust, Metal, and a packed checkpoint. Run on AC power with other
inference, builds, and GPU tests stopped. Prefix commands with
`mise exec --` if Mise is not active in your shell.

```sh
cargo xtask bench /path/to/model --dry-run
cargo xtask bench /path/to/model --build-stores
cargo xtask bench /path/to/model --cases code,prose --rounds 1
cargo xtask bench /path/to/model --output results/comparison
cargo xtask bench /path/to/model --output results/comparison --resume
```

The runner builds offline unless given `--binary`. `--build-stores` permits
missing low-bit stores to be built during load. Otherwise missing stores are
an error. Allow an extra 39 GB for Q2 and 54 GB for Q3. Loading and conversion
are excluded from prefill and decode timings.

## Suite

[suite.json](suite.json) defines 80 fresh-process samples:

- Six completion workloads × four settings × three rounds.
- One long-prefill workload and one pelican per setting.

Settings are Q4, Q4 residents with Q2 misses and cut 0.08, Q3, and Q2. All use
two adaptive drafts, an adaptive expert pool, and 8,192-token context. The mixed
setting's cut makes output timing-dependent. Configurations rotate between
rounds; pelicans run last.

Answers run to EOS. Safety caps are 4,096 tokens, except LRU, reasoning, and
pelicans at 7,168, and document summaries at 1,024. Capped or cycling answers
are saved but excluded from medians. Completion does not establish correctness.
Compare answer length and completion time alongside tg/s.

Expert precision applies to prefill and decode. The saved September 9 baseline
used Q4 batched prefill in every mode; later prefill comparisons are separate.

## Measurement

Each sample uses a fresh process without `--check`, warm repeats, prefix
caching, or developer overrides. Power is checked
at process boundaries and every 30 seconds. A power-source change invalidates the
sample; shorter transitions may be missed.

Reported GPU memory above 25 decimal GB invalidates a sample. Memory is read
after prefill scratch is released. `gpu_span_ms` is the interval between the
command buffer's GPU start and end timestamps, including gaps. `io_wait_ms`
measures host servicing of expert reads. The intervals overlap and must not
be added together. Neither measures GPU utilization.

## Reports and resume

| File | Contents |
| --- | --- |
| `report.json` | Samples, arguments, phase timings, power readings, and source/model/binary hashes |
| `README.md` | Optional run observations, preserved during regeneration |
| `summary.md` | Medians, pelicans, and links to full answers; renders on GitHub and the site |
| `outputs/` | Full generated answers |
| `pelicans/` | Unedited, XML-validated drawings |
| `gallery.html` | Local browser preview, regenerated from the report |

Browse the [saved reports](../results/README.md) on GitHub or the documentation
site. Published runs retain the report, summary, observations, outputs, and
pelicans. The HTML gallery is a local preview.

Results default to `results/<UTC timestamp>/` and are written after each sample.
New runs are ignored by Git. Use `git add -f results/<name>` to retain one.
SVG rates are reported separately; malformed or incomplete SVGs are invalid.

`--resume` requires matching binaries, model, suite, and selections. It skips
successful and content-invalid samples. Engine errors, memory-limit failures,
and power changes stop the suite and are retried on resume.

To retry capped answers, raise their caps with, for example,
`--case-cap code-lru=7168 --resume`. Completed EOS samples remain. Earlier
attempts and signatures are retained. Lower caps or other setup changes are
rejected.

Every 30 seconds, cycle detection looks for four exact repetitions of a
32–512-word block at the output tail. A match saves the evidence, ends that
sample, and continues the suite. The check does not detect every kind of loop.

Regenerate reports without inference:

```sh
cargo xtask summarize results/<name>
cargo xtask readme results/<name>
```

`summarize` preserves the run's README. `--update-readme` runs the second
command after a completed benchmark.

## Paired prefill

```sh
cargo xtask prefill --before /path/to/old/cherenkov \
  --after target/release/cherenkov --model /path/to/model \
  --out results/prefill-comparison --rounds 2
```

This compares three prompt lengths with Q4/Q3/Q2 experts and alternating
binary order. `--configs 4/2` or `--configs 4/3` selects mixed precision without
a deadline cut. Each sample generates eight tokens to check the decode handoff;
it does not measure complete answers or decode throughput.

Prepare the selected low-bit stores first. Any store build invalidates a
sample. Reports retain hashes, source diff, prompts, telemetry, output, and
power readings.
