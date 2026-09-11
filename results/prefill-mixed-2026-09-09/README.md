# Mixed-precision prefill comparison

[Timings](summary.md).

Each sample ran in a fresh process with an 8,192-token context, an adaptive pool
and two adaptive drafts. Prefix caching was disabled.

Each sample generated eight tokens to check the prefill-to-decode handoff. The
suite measures prefill throughput.

The rates exclude loading and conversion. A store build invalidates the sample.

Configurations and binary order rotate. Existing OS file caches are not purged.

The mixed configuration used 4-bit resident experts and 2-bit misses without a
deadline cut.

Low-bit prefill changes output relative to the earlier Q4-prefill
implementation.

These are two samples per cell with binary order reversed in round two. Mixed
Q4/Q3 was not timed.

All samples reported 20.98 GB Metal after prefill scratch release; this is not a
transient peak measurement.

Both binaries match those in the uniform prefill report, which records the
implementation source commit and diff. The source fields here describe the
benchmark invocation. This repository does not include the executables or
development history.

Binary hashes, phase timings, source snapshot, and power readings are in
[report.json](report.json).

Machine-specific model and executable paths in the saved reports have been
normalized to placeholders. Measurements and generated answers are unchanged.
