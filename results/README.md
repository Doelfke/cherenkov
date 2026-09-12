# Results

| Report | Scope |
| --- | --- |
| [Baseline](baseline-2026-09-09/README.md) | Complete answers, phase timings, and pelicans |
| [Low-bit prefill](prefill-lowbit-2026-09-09/README.md) | Paired Q4/Q3/Q2 prefill |
| [Mixed prefill](prefill-mixed-2026-09-09/README.md) | Paired Q4 residents with Q2 misses, without a cut |

The [benchmark suite](../benchmarks/README.md) writes to
`results/<UTC timestamp>/`, or the path given by `--output`. Smoke reports use
`results/*-smoke.json`. New results are ignored by Git; retain reviewed runs
with `git add -f results/<name>`.

Keep each run directory intact so relative output and image links resolve.
