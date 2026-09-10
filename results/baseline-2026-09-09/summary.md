# Cherenkov benchmark

Source commit: `93c514f47e0f6c234aea47c11ccdf0c847f99dac`

Fresh processes; configurations interleaved and rotated. Load is separate from
PP/s and TG/s.
Medians below include only complete, valid runs. Complete answers through EOS;
no CPU-oracle checks or prefix caching.

| Case | Configuration | Runs | Output tokens | Decode s | Load s | PP/s | TG/s | Metal GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 4-bit | 3 | 187 | 21.94 | 0.57 | 5.2 | 8.52 | 20.98 |
| code | 4-bit resident / 2-bit misses + cut 0.08 | 3 | 557 | 69.54 | 0.60 | 5.4 | 8.29 | 20.98 |
| code | 3-bit | 3 | 185 | 15.26 | 0.49 | 6.2 | 12.13 | 20.98 |
| code | 2-bit | 3 | 867 | 49.90 | 0.49 | 8.3 | 17.38 | 20.98 |
| code-lru | 4-bit | 3 | 2124 | 300.48 | 0.58 | 6.8 | 7.07 | 20.98 |
| code-lru | 4-bit resident / 2-bit misses + cut 0.08 | 3 | 1796 | 223.73 | 0.48 | 7.3 | 8.03 | 20.98 |
| code-lru | 3-bit | 3 | 2314 | 215.91 | 0.51 | 8.4 | 10.72 | 20.98 |
| code-lru | 2-bit | 3 | 4408 | 289.38 | 0.49 | 11.0 | 15.23 | 20.98 |
| debug-bisect | 4-bit | 3 | 1773 | 250.96 | 0.54 | 11.4 | 7.06 | 20.98 |
| debug-bisect | 4-bit resident / 2-bit misses + cut 0.08 | 3 | 1903 | 237.68 | 0.52 | 11.4 | 7.96 | 20.98 |
| debug-bisect | 3-bit | 3 | 1865 | 181.64 | 0.53 | 8.8 | 10.27 | 20.98 |
| debug-bisect | 2-bit | 3 | 1771 | 130.84 | 0.63 | 9.3 | 13.54 | 20.98 |
| prose | 4-bit | 3 | 692 | 89.78 | 0.57 | 5.0 | 7.71 | 20.98 |
| prose | 4-bit resident / 2-bit misses + cut 0.08 | 3 | 691 | 77.69 | 0.47 | 5.4 | 8.92 | 20.98 |
| prose | 3-bit | 3 | 461 | 46.88 | 0.47 | 6.0 | 9.83 | 20.98 |
| prose | 2-bit | 3 | 447 | 38.10 | 0.57 | 8.1 | 11.73 | 20.98 |
| reasoning | 4-bit | 3 | 4033 | 595.92 | 0.57 | 10.9 | 6.77 | 20.98 |
| reasoning | 4-bit resident / 2-bit misses + cut 0.08 | 3 | 2264 | 281.12 | 0.51 | 10.7 | 8.05 | 20.98 |
| reasoning | 3-bit | 3 | 2172 | 200.02 | 0.53 | 8.0 | 10.86 | 20.98 |
| reasoning | 2-bit | 3 | 2487 | 191.30 | 0.67 | 8.6 | 13.00 | 20.98 |
| structured | 4-bit | 3 | 372 | 47.18 | 0.57 | 13.6 | 7.88 | 20.98 |
| structured | 4-bit resident / 2-bit misses + cut 0.08 | 3 | 372 | 39.86 | 0.47 | 13.8 | 9.33 | 20.98 |
| structured | 3-bit | 3 | 372 | 29.36 | 0.47 | 11.1 | 12.67 | 20.98 |
| structured | 2-bit | 3 | 372 | 18.45 | 0.51 | 11.3 | 20.17 | 20.98 |
| prefill-long | 4-bit | 1 | 48 | 8.59 | 0.54 | 83.2 | 5.59 | 20.98 |
| prefill-long | 4-bit resident / 2-bit misses + cut 0.08 | 1 | 44 | 6.35 | 0.52 | 77.8 | 6.93 | 20.98 |
| prefill-long | 3-bit | 1 | 48 | 7.27 | 0.62 | 75.8 | 6.60 | 20.98 |
| prefill-long | 2-bit | 1 | 51 | 6.66 | 0.58 | 78.9 | 7.66 | 20.98 |
| pelican | 4-bit | 1 | 1685 | 224.85 | 0.44 | 5.3 | 7.49 | 20.98 |
| pelican | 4-bit resident / 2-bit misses + cut 0.08 | 1 | 2156 | 252.68 | 0.47 | 5.8 | 8.53 | 20.98 |
| pelican | 3-bit | 1 | 1751 | 150.64 | 0.49 | 6.6 | 11.62 | 20.98 |
| pelican | 2-bit | 1 | 2707 | 190.48 | 0.49 | 8.6 | 14.21 | 20.98 |

The mixed 4/2-bit setting includes the timing-dependent deadline cut. Its output
is not reproducible.
SVG rates are artifact-generation timings, separate from the other completion
cases.
Long-prefill token count is measured by the engine; the fixture is not claimed
to be an exact token length.
Low-bit store construction is included in load. An existing file can also
require rebuilding.
Output lengths differ across settings; compare completion times as well as TG/s.
Completion means reaching EOS; generated code and factual answers are not graded
for correctness.
Metal GB is the engine-reported allocation after prefill scratch is released,
not a measured transient peak.
Power is sampled at the start, end, and every 30 seconds; shorter transitions
can go undetected.
Telemetry times/rates have the precision printed by the engine; no load time is
included in prefill.

[Pelican gallery](gallery.html). Raw generated text is in `outputs/`; valid
extracted SVGs are in `pelicans/`.

## Run notes

- The first mixed-mode load took 75.08 s because it built the 2-bit store. That
  sample is preserved in setup_runs and outputs/setup-code-misses-2bit.txt,
  excluded from comparison medians, and rerun using the cached store. Its
  separately timed prefill was 5.98 s and decode 20.77 s.
- Both derived stores coexist with the base 4-bit store. Conversion finishes
  inside model load before either prefill or decode timing begins.
- Answers reaching the safety cap are retained as incomplete, excluded from
  completion medians, and do not stop the remaining workloads.
- The AC-to-battery transition during r3-code-misses-2bit is retained under
  previous_attempts; resume retries it on AC and excludes the interrupted
  measurement.
