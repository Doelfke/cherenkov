# Statistics

These commands query the resident server through its local control socket.
They do not load another model.

| Command | View |
| --- | --- |
| `cherenkov status` | Readiness, requests, memory, and cache usage |
| `cherenkov stats summary` | Overall expert reads, prediction, and CPU/GPU phase totals |
| `cherenkov stats layers` | Counters for each routed-expert layer |
| `cherenkov stats experts 7` | Counters for individual experts in layer 7 |
| `cherenkov dash` | Live charts, selectable rows, and details |

Add `--socket PATH` to select another server. While the model loads, `status`
reports `ready: false`; `stats` returns a not-ready error and `dash` retries.

## Summary and detailed output

`stats` prints a formatted summary by default. Layer and expert tables show
selections, cache hits and misses, prefetches, and requested reads and bytes.
Add `--json` for the complete response, including nested per-layer timing,
prediction, and precision counters:

```sh
cherenkov stats summary
cherenkov stats layers --offset 0 --limit 8
cherenkov stats experts 7 --offset 0 --limit 32
cherenkov stats layers --offset 0 --limit 8 --json > layers.json
cherenkov stats summary --json > stats.json
cherenkov stats summary > stats.md
cherenkov status --json
```

Redirected text is Markdown. Terminal output uses the terminal palette;
`NO_COLOR` disables styling. JSON retains integer byte counts and counters.

`--offset` selects the first entry; `--limit` sets the maximum returned
(default 64, range 1–128). Pages can be shorter to fit the 64 KiB control frame.
Follow `next_offset` until it is null. Each page includes its observation time;
pages queried during generation may describe different observations.

## Dashboard

`dash` refreshes once per second. The overview shows up to two minutes of
expert read throughput, CPU read waits, and GPU stage coverage. Charts use
changes between observations; row and detail panes show totals since load.
Read-volume bars are scaled to the current page. Connection failures leave
old values marked stale and gaps in the chart history.

| Key | Action |
| --- | --- |
| `1`, `2`, `3` | Overview, Layers, Experts |
| Tab | Focus the next pane |
| Arrows, `j`/`k` | Select a row or scroll details |
| Enter | Open a layer's experts; expand an expert's details |
| `f` | Expand or restore the focused pane |
| Escape | Restore a pane, return from inspection, or quit |
| `n`/`p` | Next or previous server page |
| `[`/`]` | Previous or next expert layer |
| Home/End, PgUp/PgDn | Navigate rows or details |
| `r` | Refresh now |
| `?` | Show help |
| `q`, Ctrl-C | Quit |

Selecting a row updates its detail pane. Small terminals show only the
focused pane; Tab reaches the others.

## Counter meanings

Layer IDs index the packed expert table, including MTP entries. Counters
include prefill, draft, and verification work, even when rows are later
rolled back or cut.

| Counter | Meaning |
| --- | --- |
| `selected_rows` | Rows routed to the expert |
| `cache_hits`, `cache_misses` | Resident or absent at lookup, once per expert per batch |
| `prefetch_requests` | Lookahead selections, including resident experts |
| `read_requests`, `read_bytes_requested` | Submitted reads and their requested bytes |

Each layer's `streaming` object contains `prediction`, `phases`, and `quant`.
Quant entries identify their `bits` (4, 3, or 2) and separate `prefill`,
`demand`, and `prefetch` reads. This distinguishes Q4 lookahead reads from
low-bit fallback reads in mixed mode.

Reads record requested and completed bytes, successful and failed read counts,
summed read duration, and maximum duration. Short reads count as failures;
their bytes and duration remain included. Worker read durations can overlap.

Summary rates are averages since engine load. To measure an interval, divide
changes in completed bytes or reads by the change in observation time.

| Prediction counter | Meaning |
| --- | --- |
| `target_batches` | Batches with a prediction to evaluate |
| `predicted_selected`, `predicted_unused` | Prediction outcomes |
| `selected_unpredicted` | Selected experts absent from the prediction |
| `predicted_resident` | Experts resident when predicted |
| `needed_prefetch_ready`, `needed_prefetch_late` | Required reads ready or unfinished at evaluation |

Within each quant entry, `cut_experts / eligible_weak_misses` is the cut rate.
`cut_experts / selected_experts` is the fraction of all selections cut.
`cut_batches` counts affected batches. Cuts skip computation while the read
finishes in the background.

### Prefill chunks

The summary's `prefill` object records completed batched-prefill work:
`chunks`, `tokens`, `min_chunk_tokens`, `max_chunk_tokens`, and `seconds`.
It also separates ring-reuse waits, n-gram gathering, and GPU command-buffer
spans for DeltaNet, attention, experts, and MTP. GPU spans can include waits.
Short prompts processed through the decode kernels are outside these counters.

Counters accumulate across requests and survive cancellation or rollback.
`seconds` covers completed chunks; request `prefill_seconds` also includes
checkpoint work and failed work. Expert-read bytes remain in the existing
precision-specific `prefill` read counters.

### Phase timing

Phase timing covers decode and MTP. The CPU and GPU intervals overlap.

| Field | Interval |
| --- | --- |
| `router_to_resident_seconds` | Router end to resident pass start |
| `resident_seconds` | Resident expert pass |
| `resident_to_fetched_seconds` | Resident end to fetched pass start |
| `fetched_stage_seconds` | Fetched pass through the next router boundary or final head |
| `service_wall_seconds` | CPU observes routing through releasing fetched work, including waits |
| `service_cpu_seconds` | Service-thread CPU time in that interval, including spinning |
| `prefetch_wait_seconds` | CPU wait for pending prefetch reads |
| `demand_wait_seconds` | CPU wait for required reads or the deadline decision |

CPU markers also record observation delay, preparation before resident release,
and time after release. Read-worker CPU time is outside `service_cpu_seconds`.

`gpu_stage_fraction` is compute-pass time divided by the measured GPU timeline.
`gpu_handoff_gap_fraction` is the remaining handoff time, including event and
dispatch overhead. These measure phase coverage, not shader occupancy.

`gpu_timing.status` explains timer availability; failures include an error.
`gpu_timestamps_available` indicates timer availability, and `gpu_windows` and
`invalid_gpu_windows` count samples. Fractions are null without valid samples.
The control thread reads copied statistics without waiting for the GPU.

### Requests, sessions, and memory

`status --json` includes configuration generation, active request/session IDs,
phases, token counts, timings, memory reservations, and cache/session usage.
`current` identifies the last dispatched request.

Prefill time includes prompt processing and checkpoint creation. Decode time
includes verification and enqueueing text. State switches and network writes
are excluded. Generated-token totals include partial failed responses.
Completed requests have finished writing; failed requests failed after
queueing; rejected requests failed admission.

`GET /v1/sessions/{id}` includes committed turn count, history bytes, and
cumulative `usage`: prompt, cached, and generated tokens plus prefill/decode
seconds. History and usage commit together; cancelled turns add neither.
Active-request usage is provisional until commit. Session eviction removes
its statistics.

`stats.memory` contains the latest runtime observation:

| Field | Meaning |
| --- | --- |
| `metal_allocated_bytes_observed` | Metal resource bytes |
| `mapped_weight_buffer_bytes` | Shared weight buffer length, counted once |
| `kv_index_capacity_bytes` | Reserved KV and attention-index bytes, including enabled MTP buffers |
| `context_capacity_tokens` | Engine context capacity |
| `mtp_enabled` | Whether the draft head is loaded |
| `expert_pool_bytes`, `expert_pool_slots` | Pool capacity in bytes and records |
| `device_working_set_bytes` | Metal's recommended working set |
| `allocation_limit_bytes` | Server budget after CPU reservations; null for unbounded CLI runs |
| `prefill_reserved_bytes` | Scratch allowance used when sizing the pool |
| `resident_experts` | Resident record count |

KV/index capacity excludes recurrent state, PLE history, and scratch. These
fields overlap and must not be summed. Prefix and session storage have separate
`stats.cache` and `stats.sessions` counters. Observations include uptime and
are published after load, during generation, and while idle.

CLI generation emits the same fields in a `memory_stats` JSON line on stderr.
Benchmark reports store that object in `metrics.memory`; older reports may
leave it null.
