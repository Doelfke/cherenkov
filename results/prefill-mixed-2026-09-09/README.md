# Mixed-precision prefill comparison

12 valid samples. Medians below exclude loading and conversion.

| Prompt tokens | Resident / miss bits | Before pp/s | After pp/s | Change | Pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 121 | 4 / 2 | 11.8 | 13.8 | +16.5% | 2 |
| 521 | 4 / 2 | 34.1 | 43.1 | +26.2% | 2 |
| 3221 | 4 / 2 | 107.7 | 122.5 | +13.7% | 2 |

Fresh processes, max_ctx=8192, adaptive pool, two adaptive drafts, no prefix
cache.
Exactly eight decode tokens; this suite measures prefill, not complete answers
or decode throughput.
Loading/conversion excluded; any store build invalidates the sample.
Power sampled at process boundaries; brief intervening changes may be missed.
Configurations and binary order rotate. Existing OS file caches are not purged.
No deadline cut. Mixed mode uses --experts 4 with --miss-experts 2 or 3.
Low-bit prefill changes output relative to the earlier Q4-prefill
implementation.
These are two samples per cell with binary order reversed in round two. Mixed
Q4/Q3 was not timed.
All samples reported 20.98 GB Metal after prefill scratch release; this is not a
transient peak measurement.
Both binaries match those in the [uniform prefill report](../prefill-lowbit-2026-09-09/report.json).
That report records the implementation source commit/diff; the source fields
here describe the benchmark invocation. Executables and development history
are not included in this repository.

Binary hashes, phase timings, source snapshot, and power readings are in
[report.json](report.json).

Machine-specific model and executable paths in the saved reports have been
normalized to placeholders. Measurements and generated answers are unchanged.
