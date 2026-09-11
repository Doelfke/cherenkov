# Low-bit prefill comparison

The report contains 36 valid samples. The medians exclude loading and conversion.

| Prompt tokens | Resident / miss bits | Before pp/s | After pp/s | Change | Pairs |
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

## Output checks

- Round 1, 121 tokens: the Q4 outputs are identical.
- Round 1, 521 tokens: the Q4 outputs are identical.
- Round 1, 3221 tokens: the Q4 outputs are identical.
- Round 2, 121 tokens: the Q4 outputs are identical.
- Round 2, 521 tokens: the Q4 outputs are identical.
- Round 2, 3221 tokens: the Q4 outputs are identical.

Binary hashes, timings, source and power readings are in [report.json](report.json).

[Run observations](README.md).
