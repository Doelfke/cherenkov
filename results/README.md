# Results

- [Standalone baseline](baseline-2026-09-09/README.md): full-answer workloads,
  separate load/prefill/decode timings, generated output, and pelicans.
- [Uniform low-bit prefill](prefill-lowbit-2026-09-09/README.md): paired
  Q4/Q3/Q2 prefill measurements before and after compressed expert reads.
- [Mixed prefill](prefill-mixed-2026-09-09/README.md): paired Q4-resident,
  Q2-miss measurements without a deadline cut.

Run the [benchmark suite](../benchmarks/README.md) for current measurements.
All benchmark outputs and smoke reports live here. The default benchmark
path is `results/<UTC timestamp>/`; `--output results/<name>` gives a run
a stable name. Smoke checks write `results/*-smoke.json`.

New results are ignored by Git. The baselines above are already tracked;
use `git add -f results/<name>` to save another reviewed run. Reports keep
relative output/image links, so each run directory can be moved as a unit.
The saved baselines replace machine-specific invocation paths with placeholders.
