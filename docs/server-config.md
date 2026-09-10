# Server configuration and control

The resident process owns one effective configuration. The same binary
queries it through a Unix-domain socket:

```sh
cherenkov serve /path/to/model --config cherenkov.example.toml
cherenkov status
cherenkov status --json
cherenkov config show
cherenkov config reload
```

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
usage-stream defaults. Every HTTP request captures its configuration
before reading its body; queued/active requests retain that generation.
Request JSON can override output length, streaming and usage defaults,
subject to the output and context limits. EOS policy remains server-owned.

Changes to `[server]`, `[limits]` or `[experts]` reject the entire reload
with a restart-required error. An unchanged reload does not increment
the generation. Reload without a startup config file returns an error.
`--repack` is a one-time startup action and is never repeated by reload.

Sampling, effort, independently sized resident sequences, retained session
IDs and concurrent generation are not implemented. Their configuration
fields are rejected. The current engine runs one generation at a
time; the queue/readers settings do not increase active sequence count.

## Memory and expert policy

`limits.memory_gb` bounds Metal allocation plus the reserved prefix cache,
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
two-second input deadline. Operations are `status`, `config_show` and
`config_reload`. The CLI checks errors and returns a nonzero exit status.

## Statistics

`status --json` returns a bounded snapshot, including readiness during
model load, active/queued/completed/failed/rejected request counts,
current phase/config generation/token count, cumulative generated/prompt/
cached tokens, prefill/decode seconds, and cache entries/bytes/evictions.
It stores no prompts, response history or per-request metrics history.

Prefill seconds cover cache preparation, including restore and checkpoint
work; decode seconds include synchronous stream writes. These are service
timings, not kernel benchmarks. Generated tokens include partial failed
responses. Completed requests finished successfully through the response
writer; failed requests include validation and generation failures after
queueing. Rejected counts cover queue and reader capacity rejection.

Memory reports the most recent engine observation, with its uptime
timestamp: Metal allocation, expert pool bytes, and resident expert count.
Observations are published after load, prefill and each request, and while
idle. They are not a transient peak measurement or process RSS. The
control thread never touches live GPU buffers and remains queryable while
generation runs. The `capabilities` object reports current feature limits.
