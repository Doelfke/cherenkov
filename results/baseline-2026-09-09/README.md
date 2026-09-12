# Baseline: September 9, 2026

This baseline contains 80 complete samples from a 32 GB M4 MacBook Air at
revision `93c514f`, using `Sawfwair/Qwen3.8-Flash-Next-MLX-4bit`.

- [Timings, pelicans, and full answers](summary.md)
- [Raw data and hashes](report.json)
- [Benchmark method](../../benchmarks/README.md)

## Setup

All runs used fresh processes, two adaptive drafts, an adaptive expert pool,
and 8,192-token context. No CPU checks, prefix caching, or developer overrides
were enabled. Memory was 20.98 GB after prefill scratch release.

Batched prefill used Q4 in every mode. The mixed setting also used
`--cut-weak 0.08`, so its result measures both precision and a timing-dependent
cut. All 18 workload/precision groups without the cut repeated byte-identically.

## Exclusions

- The first mixed run built the Q2 store during its 75.08-second load. It was
  saved under `setup_runs` and excluded from medians. Prefill and decode were
  timed separately at 5.98 and 20.77 seconds.
- A battery transition invalidated `r3-code-misses-2bit`. It was retried on AC;
  the failed sample remains in `previous_attempts`.
- Capped answers were excluded. Final caps were 7,168 for LRU, reasoning, and
  pelicans. Earlier attempts and arguments remain in the report.

## Interpretation

Compare completion time and answer length alongside token rates. Outputs are
unedited and were not graded for correctness. The figures apply to the saved
executable; its hashes are in the report. That executable and its development
history are not distributed here. Local paths were replaced with placeholders.
