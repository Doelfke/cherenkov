# Mixed-precision prefill comparison

[Timings](summary.md) cover 12 valid samples comparing Q4 prefill with Q4
residents and Q2 misses. Rates exclude loading and conversion.

Each sample ran on AC in a fresh process with an 8,192-token context, an
adaptive pool, and two drafts. Prefix caching and deadline cuts were disabled.
Each sample generated eight tokens to check the decode handoff. Binary order
reversed in round two. Q4/Q3 was not measured.
Memory was 20.98 GB after prefill scratch release.

The binaries match the [uniform comparison](../prefill-lowbit-2026-09-09/README.md),
whose report records the implementation revision. Low-bit prefill changes output.
[report.json](report.json) contains hashes, timings, and power readings.
