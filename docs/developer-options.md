# Developer environment reference

These are escape hatches, not normal CLI options.
Unset all of them for normal inference and reproducibility comparisons.
Diagnostic stage skipping or fake reads produce invalid model outputs.

| Variable | Value / default | Purpose |
| --- | --- | --- |
| `CHERENKOV_FAKE` | `experts`; off | Skip real expert reads for attribution; invalid numerics. |
| `CHERENKOV_SKIP` | comma-separated `experts,shared,lmhead,mixer`; empty | Skip selected decode stages; invalid numerics. |
| `CHERENKOV_LAYERS` | integer; all | Cap decoder blocks for debugging. |
| `CHERENKOV_TRACE` | presence; off | Print per-step timing and residency detail. |
| `CHERENKOV_SPIN` | `0` to block; spinning default | Shared-event waiting policy. |
| `CHERENKOV_LOOKAHEAD` | `0` to disable; enabled | One-block router lookahead and reads. |
| `CHERENKOV_ROWS_MAX` | integer; 4 | Cap rows in the short-prompt prefill path. |
| `CHERENKOV_POOL` | `set`; copy pool default | Residency-set pool over mapped experts; requires 4-bit throughout. |
| `CHERENKOV_PREFILL_MIN` | integer; 64 | Prompt length selecting the batched prefill engine. |
| `CHERENKOV_PREFILL_CHUNK` | positive integer; adaptive | Prefill rows per chunk; capped at 1,024 under `--check`. |
| `CHERENKOV_DUMP_ARGMAX` | output path; off | Per-prompt-row position, argmax, and maximum logit. |
| `CHERENKOV_DUMP_LOGITS` | output path; off | Last prompt row's f32 little-endian logits. |
| `CHERENKOV_DUMP_LA` | output path; off | JSON lookahead predictions and outcomes. |
| `CHERENKOV_DUMP_EXPERTS` | output path; off | JSON expert history. |
| `CHERENKOV_DUMP_ROUTES` | output path; off | JSON route history. |
| `CHERENKOV_DUMP_STATES` | output path; off | Router states: JSON header plus binary records. |
| `CHERENKOV_DUMP_TOKENS` | output path; off | JSON prompt and generated token IDs. |

`CHERENKOV_MODEL_DIR` selects a packed model directory for the real-weight
tests. The tests otherwise use the default managed model under the XDG
data directory and skip their bodies when the packed model is absent. Other kernel
and CPU unit tests do not require a checkpoint.
