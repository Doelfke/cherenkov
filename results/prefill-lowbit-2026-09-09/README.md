# Low-bit prefill comparison

36 valid samples. Medians below exclude loading and conversion.

| Prompt tokens | Expert bits | Before pp/s | After pp/s | Change | Pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 121 | 4 | 11.3 | 11.9 | +5.8% | 2 |
| 121 | 3 | 8.9 | 14.8 | +65.7% | 2 |
| 121 | 2 | 9.5 | 20.4 | +114.2% | 2 |
| 521 | 4 | 34.2 | 34.2 | +0.0% | 2 |
| 521 | 3 | 30.1 | 42.0 | +39.9% | 2 |
| 521 | 2 | 30.2 | 55.0 | +82.0% | 2 |
| 3221 | 4 | 97.4 | 95.2 | -2.3% | 2 |
| 3221 | 3 | 108.4 | 117.2 | +8.1% | 2 |
| 3221 | 2 | 100.1 | 108.2 | +8.1% | 2 |

Fresh processes, max_ctx=8192, adaptive pool, two adaptive drafts, no prefix
cache.
Exactly eight decode tokens; this suite measures prefill, not complete answers
or decode throughput.
Loading/conversion excluded; any store build invalidates the sample.
Power sampled at process boundaries; brief intervening changes may be missed.
Configurations and binary order rotate. Existing OS file caches are not purged.
Low-bit prefill changes output relative to the earlier Q4-prefill
implementation.
Each table cell has two samples. Long-prompt timings varied across the run; Q2
paired gains were +1.0% and +15.9%, so its long-prompt gain is less certain than
the short/medium gains.
Q4 compute is unchanged; its timing differences are control variation, not an
intended optimization.
All 36 samples were on AC at both checks, built no stores, and reported 20.98 GB
Metal after prefill scratch release. This is not a transient peak measurement.
Binary hashes and the development source commit/diff are recorded in the
report. Archived executable paths refer to the original measurement machine;
the executables and development history are not included in this repository.

## Output checks

- Round 1, 121 tokens: Q4 output identical.
- Round 1, 521 tokens: Q4 output identical.
- Round 1, 3221 tokens: Q4 output identical.
- Round 2, 121 tokens: Q4 output identical.
- Round 2, 521 tokens: Q4 output identical.
- Round 2, 3221 tokens: Q4 output identical.

Binary hashes, phase timings, source snapshot, and power readings are in
[report.json](report.json).

Machine-specific model and executable paths in the saved reports have been
normalized to placeholders. Measurements and generated answers are unchanged.
