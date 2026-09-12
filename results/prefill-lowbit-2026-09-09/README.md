# Low-bit prefill comparison

[Timings and output checks](summary.md) cover 36 valid samples comparing Q4
prefill with per-expert low-bit prefill. Rates exclude loading and conversion.

## Setup

Each sample ran on AC in a fresh process with an 8,192-token context, an
adaptive pool, and two drafts. Prefix caching and deadline cuts were disabled.
Each sample generated eight tokens to check the decode handoff. Binary order
alternated between rounds; no stores were built during measurement.
Memory was 20.98 GB after prefill scratch release.

## Results

Q4 output matched in all six pairs. Q4 computation was unchanged, so its rate
variation is a control. Low-bit prefill changes output. Long-prompt Q2 gains
varied from 1.0% to 15.9% across the two pairs; that result is less certain
than the short-prompt gains.

[report.json](report.json) contains hashes, timings, power readings, and source
metadata. Archived executable paths refer to the measurement machine.
