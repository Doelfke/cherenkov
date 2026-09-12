# Server configuration

```sh
cherenkov serve /path/to/model --config cherenkov.example.toml
cherenkov status --json
cherenkov dash
cherenkov stats summary
cherenkov stats layers
cherenkov stats experts 0 --offset 0 --limit 64
cherenkov config show
cherenkov config reload
```

See [statistics](stats.md) for summaries, detailed JSON, and dashboard controls.

[The example TOML](../cherenkov.example.toml) lists all settings.
Precedence, from lowest to highest: built-in defaults, TOML, explicit CLI flags.
Unknown keys and invalid combinations are errors.

Without `--config`, the server reads the default file described in
[storage](storage.md). `server.model_dir` replaces the positional model path;
if both are omitted, the managed model is used. Relative TOML paths resolve
from the config file's directory. Relative CLI paths resolve from the working
directory. TOML paths do not expand `~` or shell variables.

`serve --print-config` prints resolved settings without loading the model.
Explicit CLI values override TOML even when equal to built-in defaults.
Use `--pool-gb adaptive` or `--no-eos=false` to restore those defaults.

## Reload

Only `[defaults]` and `[defaults.sampling]` are reloadable. Changes to
`[server]`, `[limits]`, or `[experts]` require a restart and reject the whole
reload. Reload rereads the startup file, reapplies startup CLI flags, and
validates before replacing the configuration. It requires a startup config
file. An unchanged reload keeps the same generation number.

Requests capture defaults before their bodies are read. Queued and active
requests keep that snapshot. Request fields can override generation defaults
within server limits; EOS policy stays server-owned. Sessions retain their
committed sampling settings across reloads. `--repack` runs only at startup.

## Memory and scheduling

| Setting in `[limits]` | Default | Purpose |
| --- | --- | --- |
| `memory_gb` | 25 | Decimal GB for GPU buffers and reserved cache/session memory; maximum 25 |
| `context_tokens` | 2048 | Shared GPU context capacity |
| `prefix_cache_mib` | 512 | Prefix checkpoint budget; maximum 2048 MiB |
| `cache_max_entries` | 16 | Prefix checkpoint count |
| `cache_idle_seconds` | 900 | Prefix idle expiry; 0 disables expiry |
| `active_requests` | 2 | Requests progressing in turn on the GPU |
| `active_state_mib` | 1024 | Active checkpoints and request workspace |
| `prefill_quantum` | 128 | Prompt tokens processed before yielding |
| `max_sessions` | 16 | Retained conversations; 0 disables retention |
| `session_history_mib` | 16 | Retained history budget |
| `session_idle_seconds` | 900 | Session idle expiry; 0 disables expiry |
| `queued_requests` | 8 | Waiting requests |
| `http_readers` | 32 | Concurrent request readers |
| `request_bytes` | 4194304 | Request body and combined session input limit |
| `response_bytes` | 4194304 | Generated text limit per request |
| `max_output_tokens` | 262144 | Output ceiling, also limited by remaining context |

`memory_gb` excludes general process overhead. Buffer allocations are checked
at load and during prefill. Expert-pool sizing reserves 1 GB for prefill;
requests fail if their scratch cannot fit.

The worker runs one prefill chunk or complete decode step at a time.
Switching requests copies sequence state to and from CPU memory. Weights,
expert slots, and scratch remain shared. One active request avoids these copies.
Smaller prefill chunks reduce scheduling and cancellation latency at the cost
of more overhead.

Admission reserves checkpoint, sampling, token, and text storage for the
requested context. Requests wait if they cannot fit together; one that cannot
fit alone returns 503. A lower request context reduces this reservation but
does not resize the GPU context buffers.

Prefix checkpoints and idle sessions use independent LRU eviction and expiry.
Cache hits refresh expiry; successful turns refresh session retention. Queued,
active, and publishing turns pin their sessions. Failed or cancelled turns
preserve committed history and RNG state. Oversized prefix entries are skipped.
Expiry runs before lookup and once per idle second; GPU work can delay release.
Neither store spills to disk.

Each response writer has a 16-frame queue and a 30-second write timeout.
Overflow or write failure cancels unfinished generation. Request IDs remain
reserved through final output, bounding writer count. Queued requests can be
cancelled before admission.

## Expert policy

`resident_bits` selects 4, 3, or 2. `miss_bits` follows it unless specified;
mixed mode requires 4-bit residents. The policy applies to routed experts in
prefill and decode. Dense and shared-expert precision is unchanged. One low-bit
layout is attached per engine, even when both variants exist on disk.

`build_missing_store=true` permits startup conversion. Set it to false to
require a valid existing store. Precision never changes automatically on
memory or IO failure. An infeasible explicit pool fails startup.

`cut_weak` is off by default. A nonzero value permits skipping late weak
experts and makes output depend on IO timing.
[Developer overrides](developer-options.md) apply independently of TOML.

## Control socket

The default is `TMPDIR/cherenkov-UID/control.sock`. Server and CLI must use the
same TMPDIR, or select an instance with `--socket /absolute/path/control.sock`.
The server creates its private directory if needed; its parent must exist.

The directory must be owned by the server user with mode `0700`; the socket
uses `0600`. Both peers check the effective UID with `getpeereid`. Processes
running as that user are trusted. A held lock prevents duplicate owners and
permits stale socket cleanup. Symlinks, unsafe permissions, and live sockets
are rejected.

Control is separate from HTTP. Each connection carries one newline-terminated
JSON request and response, limited to 64 KiB with a two-second input deadline.
Operations are `status`, `stats_summary`, `stats_layers`, `stats_experts`,
`config_show`, and `config_reload`. CLI errors return a nonzero exit status.
