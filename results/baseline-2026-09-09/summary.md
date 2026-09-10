# Cherenkov benchmark

Source commit: `93c514f47e0f6c234aea47c11ccdf0c847f99dac`

| Case | Configuration | Runs | Output tokens | Decode s | Load s | pp/s | tg/s | Metal GB |
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

[Run observations](README.md).

## Pelicans

These are unedited model outputs from the benchmark.

| 4-bit | 4-bit / 2-bit misses + cut |
| --- | --- |
| ![Pelican](pelicans/exact-4bit.svg) | ![Pelican](pelicans/misses-2bit.svg) |

| 3-bit | 2-bit |
| --- | --- |
| ![Pelican](pelicans/all-3bit.svg) | ![Pelican](pelicans/all-2bit.svg) |

## Answers

| Sample | Status |
| --- | --- |
| [r1-code-exact-4bit](outputs/r1-code-exact-4bit.txt) | ok |
| [r1-code-all-3bit](outputs/r1-code-all-3bit.txt) | ok |
| [r1-code-all-2bit](outputs/r1-code-all-2bit.txt) | ok |
| [r1-code-lru-exact-4bit](outputs/r1-code-lru-exact-4bit.txt) | ok |
| [r1-code-lru-misses-2bit](outputs/r1-code-lru-misses-2bit.txt) | ok |
| [r1-code-lru-all-3bit](outputs/r1-code-lru-all-3bit.txt) | ok |
| [r1-code-misses-2bit](outputs/r1-code-misses-2bit.txt) | ok |
| [r1-debug-bisect-exact-4bit](outputs/r1-debug-bisect-exact-4bit.txt) | ok |
| [r1-code-lru-all-2bit](outputs/r1-code-lru-all-2bit.txt) | ok |
| [r1-debug-bisect-misses-2bit](outputs/r1-debug-bisect-misses-2bit.txt) | ok |
| [r1-debug-bisect-all-3bit](outputs/r1-debug-bisect-all-3bit.txt) | ok |
| [r1-debug-bisect-all-2bit](outputs/r1-debug-bisect-all-2bit.txt) | ok |
| [r1-prose-exact-4bit](outputs/r1-prose-exact-4bit.txt) | ok |
| [r1-prose-misses-2bit](outputs/r1-prose-misses-2bit.txt) | ok |
| [r1-prose-all-3bit](outputs/r1-prose-all-3bit.txt) | ok |
| [r1-prose-all-2bit](outputs/r1-prose-all-2bit.txt) | ok |
| [r1-reasoning-exact-4bit](outputs/r1-reasoning-exact-4bit.txt) | ok |
| [r1-reasoning-misses-2bit](outputs/r1-reasoning-misses-2bit.txt) | ok |
| [r1-reasoning-all-3bit](outputs/r1-reasoning-all-3bit.txt) | ok |
| [r1-reasoning-all-2bit](outputs/r1-reasoning-all-2bit.txt) | ok |
| [r1-structured-exact-4bit](outputs/r1-structured-exact-4bit.txt) | ok |
| [r1-structured-misses-2bit](outputs/r1-structured-misses-2bit.txt) | ok |
| [r1-structured-all-3bit](outputs/r1-structured-all-3bit.txt) | ok |
| [r1-structured-all-2bit](outputs/r1-structured-all-2bit.txt) | ok |
| [r1-prefill-long-exact-4bit](outputs/r1-prefill-long-exact-4bit.txt) | ok |
| [r1-prefill-long-misses-2bit](outputs/r1-prefill-long-misses-2bit.txt) | ok |
| [r1-prefill-long-all-3bit](outputs/r1-prefill-long-all-3bit.txt) | ok |
| [r1-prefill-long-all-2bit](outputs/r1-prefill-long-all-2bit.txt) | ok |
| [r2-code-misses-2bit](outputs/r2-code-misses-2bit.txt) | ok |
| [r2-code-all-3bit](outputs/r2-code-all-3bit.txt) | ok |
| [r2-code-all-2bit](outputs/r2-code-all-2bit.txt) | ok |
| [r2-code-exact-4bit](outputs/r2-code-exact-4bit.txt) | ok |
| [r2-code-lru-misses-2bit](outputs/r2-code-lru-misses-2bit.txt) | ok |
| [r2-code-lru-all-3bit](outputs/r2-code-lru-all-3bit.txt) | ok |
| [r2-code-lru-all-2bit](outputs/r2-code-lru-all-2bit.txt) | ok |
| [r2-code-lru-exact-4bit](outputs/r2-code-lru-exact-4bit.txt) | ok |
| [r2-debug-bisect-misses-2bit](outputs/r2-debug-bisect-misses-2bit.txt) | ok |
| [r2-debug-bisect-all-3bit](outputs/r2-debug-bisect-all-3bit.txt) | ok |
| [r2-debug-bisect-all-2bit](outputs/r2-debug-bisect-all-2bit.txt) | ok |
| [r2-debug-bisect-exact-4bit](outputs/r2-debug-bisect-exact-4bit.txt) | ok |
| [r2-prose-misses-2bit](outputs/r2-prose-misses-2bit.txt) | ok |
| [r2-prose-all-3bit](outputs/r2-prose-all-3bit.txt) | ok |
| [r2-prose-all-2bit](outputs/r2-prose-all-2bit.txt) | ok |
| [r2-prose-exact-4bit](outputs/r2-prose-exact-4bit.txt) | ok |
| [r2-reasoning-misses-2bit](outputs/r2-reasoning-misses-2bit.txt) | ok |
| [r2-reasoning-all-3bit](outputs/r2-reasoning-all-3bit.txt) | ok |
| [r2-reasoning-all-2bit](outputs/r2-reasoning-all-2bit.txt) | ok |
| [r2-reasoning-exact-4bit](outputs/r2-reasoning-exact-4bit.txt) | ok |
| [r2-structured-misses-2bit](outputs/r2-structured-misses-2bit.txt) | ok |
| [r2-structured-all-3bit](outputs/r2-structured-all-3bit.txt) | ok |
| [r2-structured-all-2bit](outputs/r2-structured-all-2bit.txt) | ok |
| [r2-structured-exact-4bit](outputs/r2-structured-exact-4bit.txt) | ok |
| [r3-code-all-3bit](outputs/r3-code-all-3bit.txt) | ok |
| [r3-code-all-2bit](outputs/r3-code-all-2bit.txt) | ok |
| [r3-code-exact-4bit](outputs/r3-code-exact-4bit.txt) | ok |
| [r3-code-misses-2bit](outputs/r3-code-misses-2bit.txt) | ok |
| [r3-code-lru-all-3bit](outputs/r3-code-lru-all-3bit.txt) | ok |
| [r3-code-lru-all-2bit](outputs/r3-code-lru-all-2bit.txt) | ok |
| [r3-code-lru-exact-4bit](outputs/r3-code-lru-exact-4bit.txt) | ok |
| [r3-code-lru-misses-2bit](outputs/r3-code-lru-misses-2bit.txt) | ok |
| [r3-debug-bisect-all-3bit](outputs/r3-debug-bisect-all-3bit.txt) | ok |
| [r3-debug-bisect-all-2bit](outputs/r3-debug-bisect-all-2bit.txt) | ok |
| [r3-debug-bisect-exact-4bit](outputs/r3-debug-bisect-exact-4bit.txt) | ok |
| [r3-debug-bisect-misses-2bit](outputs/r3-debug-bisect-misses-2bit.txt) | ok |
| [r3-prose-all-3bit](outputs/r3-prose-all-3bit.txt) | ok |
| [r3-prose-all-2bit](outputs/r3-prose-all-2bit.txt) | ok |
| [r3-prose-exact-4bit](outputs/r3-prose-exact-4bit.txt) | ok |
| [r3-prose-misses-2bit](outputs/r3-prose-misses-2bit.txt) | ok |
| [r3-reasoning-all-3bit](outputs/r3-reasoning-all-3bit.txt) | ok |
| [r3-reasoning-all-2bit](outputs/r3-reasoning-all-2bit.txt) | ok |
| [r3-reasoning-exact-4bit](outputs/r3-reasoning-exact-4bit.txt) | ok |
| [r3-reasoning-misses-2bit](outputs/r3-reasoning-misses-2bit.txt) | ok |
| [r3-structured-all-3bit](outputs/r3-structured-all-3bit.txt) | ok |
| [r3-structured-all-2bit](outputs/r3-structured-all-2bit.txt) | ok |
| [r3-structured-exact-4bit](outputs/r3-structured-exact-4bit.txt) | ok |
| [r3-structured-misses-2bit](outputs/r3-structured-misses-2bit.txt) | ok |
| [r1-pelican-exact-4bit](outputs/r1-pelican-exact-4bit.txt) | ok |
| [r1-pelican-misses-2bit](outputs/r1-pelican-misses-2bit.txt) | ok |
| [r1-pelican-all-3bit](outputs/r1-pelican-all-3bit.txt) | ok |
| [r1-pelican-all-2bit](outputs/r1-pelican-all-2bit.txt) | ok |
