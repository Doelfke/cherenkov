# Low-bit prefill comparison

The report contains 12 valid samples. The medians exclude loading and conversion.

| Prompt tokens | Resident / miss bits | Before pp/s | After pp/s | Change | Pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 121 | 4 / 2 | 11.8 | 13.8 | +16.5% | 2 |
| 521 | 4 / 2 | 34.1 | 43.1 | +26.2% | 2 |
| 3221 | 4 / 2 | 107.7 | 122.5 | +13.7% | 2 |

Binary hashes, timings, source and power readings are in [report.json](report.json).

[Run observations](README.md).
