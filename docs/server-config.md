# Server configuration and control

The resident process owns one effective configuration. The same binary
queries it through a Unix-domain socket:

```sh
cherenkov serve /path/to/model --config cherenkov.example.toml
cherenkov status
cherenkov status --json
cherenkov dash
cherenkov stats summary
cherenkov stats layers
cherenkov stats experts 0 --offset 0 --limit 64
cherenkov config show
cherenkov config reload
```

`dash` refreshes once per second. The overview plots expert read throughput,
CPU read waits, and GPU stage coverage over up to two minutes. Charts use
changes between observations; tables show totals since model load. GPU
coverage is stage time divided by measured GPU phase time. Read-volume bars
scale to the current page.

Use `1`/`2`/`3` for Overview, Layers, or Experts; Tab to focus a pane; and
arrows or `j`/`k` to select rows or scroll details. Enter opens a layer's
experts, `f` expands a pane, and Escape returns. `n`/`p` changes pages and
`[`/`]` changes the expert layer. `?` shows all keys; `q` or Ctrl-C exits.
Small terminals show the focused pane. `--socket PATH` selects another
server. The dashboard is read-only and retries after connection errors,
leaving gaps in chart history.

Use [the example TOML](../cherenkov.example.toml) as the schema reference.
`server.model_dir` can replace the positional model directory. When both are
omitted, the server uses the pinned managed model from [the data
store](storage.md). Relative
model/socket paths in the file are relative to the file's directory;
relative CLI paths are relative to the starting working directory.
Paths do not expand shell variables or `~` inside TOML.

Precedence is built-in defaults, then the selected TOML file, then explicitly
supplied CLI flags. Without `--config`, the server reads the one standard
`~/.config/cherenkov/cherenkov.toml` path if it exists (or under
`XDG_CONFIG_HOME` when set to an absolute path). There is no directory search.
`--root` relocates data, scratch and this default config file; `[server].root`
can relocate server model lookup, and an explicit CLI root takes precedence.
Unknown sections/keys and invalid combinations fail validation.
`serve --print-config` prints resolved TOML without
loading weights, building stores, or binding listeners. Explicit values that
match built-in defaults still override the file. Use `--pool-gb adaptive`
to reset a configured pool budget, or `--no-eos=false` to restore EOS stopping.

## Reload

`config reload` rereads the original file and reapplies the original CLI
overrides. It validates the complete candidate before publishing anything.
Only `[defaults]` is reloadable: output length, EOS handling, streaming and
usage-stream defaults, and `[defaults.sampling]`. Every HTTP request captures
its configuration
before reading its body; queued/active requests retain that generation.
Request JSON can override sampling, output length, streaming and usage defaults,
subject to the output and context limits. EOS policy remains server-owned.

Changes to `[server]`, `[limits]` or `[experts]` reject the entire reload
with a restart-required error. An unchanged reload does not increment
the generation. Reload without a startup config file returns an error.
`--repack` is a one-time startup action and is never repeated by reload.

Retained sessions capture sampling defaults when created and retain their
committed settings across reloads. New stateless requests use the latest
captured configuration.

## Memory and expert policy

`limits.memory_gb` bounds Metal allocation plus reserved prefix cache, active
sequence workspace and session history,
up to 25 decimal GB. It is not a whole-process physical-memory cap. HTTP
bodies and queues have separate bounds; JSON decoding has additional CPU
overhead. The remaining unified memory and OS file cache must still fit
the machine. The engine applies its buffer allocation guard during load
and subsequent scratch allocation, and reserves 1 GB of headroom when
checking the expert pool against this cap. A prompt whose scratch still
cannot fit fails rather than exceeding the allocation limit.

`prefix_cache_mib` reserves up to 2048 MiB. The cache evicts least recently
used checkpoints to meet its byte and entry limits. Hits refresh idle
expiry; `cache_idle_seconds=0` disables time-based expiry. Expiry runs
before lookup and once per second while the engine is idle. Busy GPU work
can delay physical reclamation until a safe engine boundary. Oversized
entries are skipped. This cache stores hybrid model state, not responses
or persistent conversation history.

`active_requests=2` permits round-robin progress, one GPU operation at a time.
`prefill_quantum=128` caps each prompt chunk. Smaller chunks improve scheduling
and cancellation responsiveness but can increase prefill overhead. Decode yields
after a complete verification step. A single active request avoids checkpoint
copies between steps; switching requests copies hybrid state to/from CPU
vectors.
The expert pool and execution scratch remain shared.

`active_state_mib=1024` reserves active checkpoint storage, sampler workspace,
token history and bounded text copies. Admission estimates the requested prompt
plus output and speculative headroom, including vector growth. A request that
cannot fit alone returns 503; requests that cannot fit together wait. This
reservation is separate from prefix-cache snapshots. GPU context buffers are
allocated for the server maximum; smaller `context_tokens` requests impose a
logical cap and reduce checkpoint reservations, without resizing those buffers.

`max_sessions=16`, `session_history_mib=16` and `session_idle_seconds=900` bound
retained conversations. Zero sessions disables retention; zero idle seconds
disables expiry. History evicts least recently used idle sessions. Queued,
active
and publishing turns pin their sessions. Successful turns refresh retention;
cancelled or failed turns leave prior history/settings/RNG intact. History plus
new messages must also fit `request_bytes`. There is no disk spill.

`response_bytes=4194304` caps each generated text. Each response writer has a
16-frame queue; overflow or a failed write cancels its generation. Writer
sockets
have a 30-second write timeout. Registered request IDs remain bounded through
final output, so stalled writers cannot create unlimited threads. Requests
waiting
in the queue can be cancelled without waiting for a GPU slot.

`resident_bits` selects 4, 3 or 2 bits. Omitted `miss_bits` follows it;
4-bit residents may instead fetch misses at 3 or 2 bits. Both cached
low-bit stores may exist on disk, but one low-bit layout is attached to
the engine. The policy applies to routed experts in prefill and decode;
dense and shared-expert weights retain their existing precision.

There is no automatic precision reduction on memory or IO failure.
`build_missing_store=false` requires an existing validated low-bit store;
true allows construction at startup. `cut_weak` is a separate opt-in
deadline policy that can skip late weak expert contributions and makes
output depend on IO timing. An explicit infeasible pool fails startup.

Developer environment escape hatches remain available and can affect
execution independently of TOML; unset them for normal serving. They are
documented in [developer-options.md](developer-options.md).

## Control socket and authentication

The default socket is `TMPDIR/cherenkov-UID/control.sock` (using Rust's
system temporary directory). Server and CLI must use the same TMPDIR;
pass `--socket /absolute/path/control.sock` to select an explicit instance.
The parent directory must be owned by the server user and mode `0700`;
the server creates that directory if absent. The socket is mode `0600`.
Both server and client also check the peer's effective UID with
`getpeereid`. Other processes running as the same user are trusted.

A held advisory lock prevents two servers from owning the same control
path. After a crash, startup can remove a stale, owned socket while holding
that lock. It refuses symlinks, non-socket replacements, unsafe directory
permissions and live sockets. The small lock file remains for reuse.
The parent of a newly created private directory must already exist.

Control is not exposed on the OpenAI HTTP listener. No bearer token or TLS
configuration is needed for local control. The protocol is one newline-
terminated JSON request/response per connection, limited to 64 KiB and a
two-second input deadline. Operations are `status`, `stats_summary`,
`stats_layers`, `stats_experts`, `config_show` and `config_reload`.
The CLI returns a nonzero exit status on error.

## Statistics

`status --json` returns a bounded snapshot, including readiness during
model load, active/queued/completed/cancelled/failed/rejected request counts,
current phase/config generation/token count, cumulative generated/prompt/
cached tokens, prefill/decode seconds, and cache entries/bytes/evictions.
It includes active request/session IDs, per-request phase, token counts and
timings, active-state reservations, and session history bytes/count/evictions. The
`current` field identifies the most recently dispatched request. Statistics
store no prompts, response history or per-request metrics history.

Prefill seconds cover advancing prompt chunks and storing checkpoints; decode
seconds cover verification steps and enqueuing text. State switches and network
writer time are excluded. These are worker timings, not end-to-end latency.
Generated tokens include partial failed
responses. Completed requests finished successfully through the response
writer; failed requests include validation and generation failures after
queueing. Rejected counts cover reader, request-admission and session-capacity rejection.

`stats.memory` reports the latest runtime snapshot and its uptime timestamp:

| Field | Meaning |
| --- | --- |
| `metal_allocated_bytes_observed` | Metal resource bytes at observation time |
| `mapped_weight_buffer_bytes` | Shared weight buffer length, counted once |
| `kv_index_capacity_bytes` | Reserved KV and attention-index buffer bytes |
| `context_capacity_tokens` | Engine context capacity |
| `mtp_enabled` | Whether the draft head is loaded |
| `expert_pool_bytes` | Expert pool capacity in bytes |
| `expert_pool_slots` | Expert pool capacity in records |
| `resident_experts` | Current resident record count |

The CLI emits the same fields as a `memory_stats {JSON}` line on stderr after
generation. Benchmarks preserve the object in each run's `metrics.memory`.
Older telemetry leaves that field null. Byte counts remain integers.

KV/index capacity includes enabled MTP buffers. Recurrent state, PLE history,
and scratch buffers are outside that count. The fields describe capacities
and observations; they do not form a sum of process memory. Server prefix
and session storage remain in their separate `stats.cache` and
`stats.sessions` counters.

Observations are published after load, prefill and each request, and while
idle. They are not a transient peak measurement or process RSS. The
control thread never touches live GPU buffers and remains queryable while
generation runs. The `capabilities` object reports current feature limits.

### Sessions

`GET /v1/sessions/{id}` includes `stats`: committed turn count, retained and
reserved history bytes, and cumulative `usage`. Usage contains `prompt_tokens`,
`cached_tokens`, `generated_tokens`, `prefill_seconds` and `decode_seconds`.
Each turn counts its full rendered prompt, including reused tokens.
History and usage commit together; cancelled or unpublished turns add neither.
Eviction removes the session's statistics with its history.

Active requests expose the same usage fields in `status --json`, along with
their `reserved_state_bytes`. These are provisional until the turn commits.

### Layers and experts

`stats layers` returns totals by routed-expert layer. `stats experts LAYER`
returns that layer's individual experts. Both commands accept `--offset` and
`--limit` (default 64, maximum 128).

Stats commands show field lists and tables, with muted rules and accented headings.
The colors follow the terminal palette. Wide tables become aligned fields on
narrow terminals, and reports remain in the terminal scrollback.
`NO_COLOR` disables color. Redirected output is Markdown; `--json` returns the
full response. JSON pages include `data`, `total`, `next_offset`, and an
observation timestamp. A null `next_offset` marks the end.
Layer IDs index the packed expert table; layer names identify trunk and MTP entries.

During model loading, stats commands return a not-ready error. `status` remains
available and reports `ready: false`.

| Counter | Meaning |
| --- | --- |
| `selected_rows` | Rows routed to the expert |
| `cache_hits`, `cache_misses` | Resident or absent at lookup, once per expert per batch |
| `prefetch_requests` | Lookahead selections, including already-resident experts |
| `read_requests`, `read_bytes_requested` | Reads submitted and their requested bytes |

These counters accumulate across all requests until the server exits. They
include prefill, draft and verification work, including rows later rolled back
or skipped by the deadline policy. Read counters use the selected record's
precision and do not measure completed IO. The control thread reads a copied
snapshot; it does not synchronize with the GPU to answer a query.

### Expert streaming and phase timing

```sh
cherenkov stats summary
cherenkov stats summary > stats.md
cherenkov stats summary --json > stats.json
cherenkov stats layers --offset 0 --limit 8 --json > layers.json
```

The summary contains overall expert read rates and CPU/GPU phase totals.
Each layer's `streaming` object contains `prediction`, `phases`, and `quant`.
The quant entries identify their actual `bits` (4, 3 or 2). Read counters within
each entry separate `prefill`, `demand`, and `prefetch` traffic. Q4 prefetches
and Q2 fallback reads therefore remain distinguishable in mixed mode.

Reads report requested and completed bytes, successful and failed read counts,
summed read duration, and maximum duration. Short reads count as failures;
their transferred bytes and duration remain included. Durations measure the
read operation in its worker thread and can overlap. Bytes describe application
reads, including reads served from the OS cache.

`bytes_per_second` and `reads_per_second` are averages since engine load.
For a sampling window, divide differences in completed bytes or reads by the
difference in `elapsed_seconds`. These rates include all expert read sources.

Prediction counters describe the targeted batch, before joining its prefetches:

- `predicted_selected` and `predicted_unused` count prediction outcomes.
- `selected_unpredicted` counts selections absent from that batch's prediction.
- `needed_prefetch_ready` and `needed_prefetch_late` describe actual reads.
- `predicted_resident` counts experts already resident when predicted.
- `target_batches` counts batches with a prediction to evaluate.

Per-quant `eligible_weak_misses` counts weak misses in cut-enabled decode
batches. `cut_experts / eligible_weak_misses` gives the cut rate;
`cut_experts / selected_experts` gives the overall cut fraction.
`cut_batches` counts affected batches. A cut skips computation while its read
finishes in the background. All selection counts are per expert per batch;
`selected_rows` separately counts routed rows.

Phase timing covers decode and MTP handoffs:

| Field | Interval |
| --- | --- |
| `router_to_resident_seconds` | Router pass end to resident pass start |
| `resident_seconds` | Resident expert pass |
| `resident_to_fetched_seconds` | Resident pass end to fetched pass start |
| `fetched_stage_seconds` | Fetched pass start to the next router boundary, or final head completion |
| `service_wall_seconds` | CPU observes routing through releasing fetched work, including waits |
| `service_cpu_seconds` | Service thread CPU consumption in that interval, including spinning |
| `prefetch_wait_seconds` | CPU wait to join pending prefetch reads |
| `demand_wait_seconds` | CPU wait for required reads or the deadline decision |

CPU markers also report observation delay, preparation before resident release,
and time after resident release. These overlap the GPU phases; they are not
additional pieces of the GPU timeline. Read-worker CPU time is outside
`service_cpu_seconds`.

`gpu_handoff_gap_fraction` measures the two gaps as a fraction of the measured
GPU timeline; `gpu_stage_fraction` measures its compute-pass coverage. Gaps
include event and dispatch overhead. These fractions describe whether work
is available between phases, rather than shader occupancy. Whole-command-buffer
start/end spans include these gaps and cannot establish utilization.
The runner labels that older measurement `gpu-span`; benchmark JSON stores
it as `gpu_span_ms` and accepts telemetry with the former `gpu-active` label.

`gpu_timing.status` reports `available`, `unsupported`, `disabled`, or `failed`.
A disabled timer includes a `reason`; a failed timer includes an `error` and
logs it once at startup. Inference continues when timing is unavailable.

The engine samples existing [Metal compute-pass boundaries][metal-counters]
and resolves them
after the normal completion wait. It adds no GPU waits or encoder boundaries.
Unsupported counters or a capped diagnostic trunk set
`gpu_timestamps_available=false`; fractions stay null without valid samples.
`gpu_windows` and `invalid_gpu_windows` report sample coverage.

Responses may contain fewer entries than `--limit` to fit the control socket's
64 KiB frame. Follow `next_offset` until it is null. All counters are cumulative;
pages queried during generation can have different observation timestamps.

[metal-counters]: https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers
