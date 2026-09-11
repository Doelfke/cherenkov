# Low-bit prefill comparison

[Timings and output checks](summary.md).

Each sample ran in a fresh process with an 8,192-token context, an adaptive pool
and two adaptive drafts. Prefix caching was disabled.

Each sample generated eight tokens to check the prefill-to-decode handoff. The
suite measures prefill throughput.

The rates exclude loading and conversion. A store build invalidates the sample.

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

The report records binary hashes and the development source commit and diff.
Archived executable paths refer to the measurement machine. This repository does
not include those executables or the development history.

Binary hashes, phase timings, source snapshot, and power readings are in
[report.json](report.json).

Machine-specific model and executable paths in the saved reports have been
normalized to placeholders. Measurements and generated answers are unchanged.
