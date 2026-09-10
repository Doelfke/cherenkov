# Cherenkov baseline, September 9, 2026

This records the engine before the CPU/Metal readability cleanup, at
commit `93c514f47e0f6c234aea47c11ccdf0c847f99dac`, on a 32 GB M4 MacBook Air.
The model is `Sawfwair/Qwen3.8-Flash-Next-MLX-4bit`, packed locally for
Cherenkov. Exact model metadata and source hashes are in
[report.json](report.json).

- [Timings and pelicans in the browser](gallery.html)
- [Markdown timing report](summary.md)
- [Full generated answers](outputs/)
- [Benchmark method and commands](../../benchmarks/README.md)

## Completion

All 80 samples completed through EOS on AC, including four valid pelican
SVGs. Reported Metal allocation stayed at 20.98 GB. The 18 workload/precision
groups without a deadline cut produced byte-identical answers across all
three rounds. Power changed to battery during
`r3-code-misses-2bit`; that failed sample and its power readings are saved
under `previous_attempts`, with its answer in `attempts/`, and excluded
from medians. The sample was retried on AC. Source validation ran during
pauses between benchmark samples; retained measurements always use the
pinned executable without concurrent build or correctness checks.

## Setup

All configurations use two adaptive MTP drafts, an adaptive expert pool,
and 8,192 context capacity. Every sample starts a fresh native Metal
process and generates through EOS. OS file caches remain warm. The mixed
setting uses both 2-bit misses and the timing-dependent `--cut-weak 0.08`;
it is not a measurement of precision alone. The suite does not use CPU
oracle checks, server prefix caching, or developer environment overrides.

Measurements stay on a saved copy of the original release executable
while cleanup edits are prepared. Its SHA-256 is
`d54f24aae702db7ea21745387555c79399c588478ec592c12a58788866f865de`.
The recorded source hash was checked against the files in the commit
above. The initial dirty status reflects an untracked `.clangd`, outside
the compiled source.

## Conversion and capped attempts

The first mixed-mode sample built the 2-bit store. Its 75.08-second load
is preserved in `report.json` under `setup_runs`, with its
[original answer](outputs/setup-code-misses-2bit.txt). It is excluded from
comparison medians and was rerun with the cached store. Prefill was
separately timed at 5.98 seconds and decode at 20.77 seconds; conversion
was part of neither phase. The base, 2-bit, and 3-bit files coexist.

The first 2-bit LRU answer hit the 4,096-token ceiling while expanding
its tests. That [capped output](attempts/r1-code-lru-all-2bit-cap-4096.txt)
is preserved, and the LRU ceiling was raised to 7,168 without changing
context capacity. The retry finished naturally at 4,408 tokens. Its
prefix matches the original attempt byte for byte, excluding the
runner's final newline. The sequence advances through numbered test
cases; the repetition detector did not identify a cycle.

After round two's explanation cases, the remaining reasoning and pelican
ceilings were also raised to 7,168: the first 4-bit reasoning answer had
used 4,033 of its 4,096-token allowance. Completed EOS samples retain
their original caps. Context capacity, prompts, and executable stayed
the same.

The final overrides are `--case-cap code-lru=7168 --case-cap reasoning=7168
--case-cap pelican=7168`. Exact arguments and earlier cap signatures are
retained per sample in `report.json`.

## Reading the outputs

These are completion workloads, not equal-length throughput tests or a
model-accuracy score. Faster token generation can produce a longer
answer, so compare decode seconds and output length alongside TG/s.
Power samples and clock probes are retained to expose power changes and
the fanless machine's sustained-load behavior. Reported Metal allocation
is taken after prefill scratch is released, not a measurement of its peak.

Some first-round examples illustrate why the generated text is saved:

- The 2-bit LRU answer uses 4,408 tokens versus 2,124 for 4-bit.
- All four reasoning answers give 9.625 hours. The 2-bit answer checks a
  different allocation instead of the immediately preceding completion.
- The extraction answers wrap JSON in Markdown despite the JSON-only
  instruction. For order B08, 4-bit and 3-bit report the correct $45.30;
  this round's mixed and 2-bit answers report $46.30.

These are manual observations on individual answers, not a comprehensive
correctness evaluation. All text and extracted SVGs are kept unedited.

Machine-specific model and executable paths in the saved reports have been
normalized to placeholders. Measurements are unchanged; original paths remain
in Git history.
