# Running Cherenkov

The [README](../README.md) covers installation and the first run. This guide
covers the HTTP interface, prefix caching, precision and measurement.

## Local OpenAI-compatible server

```sh
target/release/cherenkov serve /path/to/model --port 8080 --max-ctx 4096
```

Server settings can also come from [TOML](../cherenkov.example.toml):

```sh
target/release/cherenkov serve /path/to/model --config cherenkov.example.toml
target/release/cherenkov status --json
target/release/cherenkov config show
target/release/cherenkov config reload
```

Control uses a private Unix-domain socket with same-user peer checks.
Reload applies request defaults atomically; engine/policy changes require
restart. See [configuration and control](server-config.md) for limits,
precedence, cache retention, socket selection and statistics.

Use base URL `http://127.0.0.1:8080/v1` and model `cherenkov`. The server
binds to localhost, keeps one model loaded, and serializes generation
through a bounded queue. It provides `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/completions`, and `GET /health`.
Both completion endpoints support JSON responses and SSE streaming,
finish reasons, token usage, and `stream_options.include_usage`.
An API-key placeholder is sufficient for clients that require one;
this local server does not authenticate requests.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cherenkov","messages":[{"role":"user","content":"Explain hash collisions."}],"max_tokens":128,"temperature":0,"stream":true}'
```

The API supports greedy text generation: `temperature` must be 0, `top_p`
must be 1, and `n` must be 1. These are also the defaults.
Chat accepts text messages with system/developer/user/assistant roles;
text completions accept a string prompt. `max_completion_tokens` and
`max_tokens` are supported. Sampling, stop strings, tool calling, JSON
schema constraints, images, and the Responses API are not implemented;
these unsupported generation controls return an error. Reasoning-effort
and template variables are not implemented and currently have no effect.
Requests use Content-Length; chunked request bodies
are not supported. CLI-only `--raw`, `--check`, and `--repeat` do not apply
to the server.

### Repeated-prefix cache

`--prefix-cache-mb 512` is the default; `0` disables prefix caching.
The server restores the longest matching checkpoint, then prefills only
the uncached suffix. Checkpoints include attention KV and QSA index
caches, DeltaNet recurrent state, convolution/PLE histories, token
history, and MTP state. MTP's following-token dependency is checked
before reuse. This is a state cache, not a stored-response cache.

Checkpoints cover the shared messages before the last user turn, the
message boundary before the assistant response, prompt extensions, and
complete prompts. Long prefill chunks can add checkpoints too. The cache
uses LRU eviction; entries too large for the budget are skipped. A short
checkpoint is about 113 MiB plus context-dependent state, so 512 MiB
holds roughly four short checkpoints. Long prefixes hold fewer. Entries
are in memory and disappear when the server exits.

TOML also controls the checkpoint count and idle expiry (16 entries and
900 seconds by default). Hits refresh expiry; idle housekeeping reclaims
expired state. These are checkpoints, not retained conversation sessions.

The cache budget is reserved from the adaptive expert pool. Server
buffer allocations plus the reserved prefix cache are limited to
25 decimal GB, including prefill scratch allocations. This is distinct
from total process/system memory and the OS file cache. Reused tokens
appear in `usage.prompt_tokens_details.cached_tokens`; with streaming,
request `stream_options.include_usage` for the final usage event.

Cache boundaries and warm expert residency can change floating-point
accumulation order. As with the original engine's warm repeats, cached
and fresh runs are not guaranteed to produce byte-identical text on
close argmax decisions. The checkpoint itself restores state byte for
byte. [Server validation](validation.md) includes live API,
prefix reuse, and state-restoration checks.

The [serving design](serving-design.md) describes further work on
sampling, native reasoning controls, retained conversations and concurrent
generation; those features are not yet implemented.

## Expert precision and speculation

Default: **4-bit experts, two adaptive MTP drafts, no deadline cut**.
`--drafts 0` disables speculation and does not load the draft head.
Generation uses greedy argmax and the original no-thinking ChatML
wrapper; `--raw` supplies the prompt without that wrapper.

```sh
# Lower-precision experts in prefill and decode:
target/release/cherenkov /path/to/model 'Explain hash collisions.' --experts 3
# Keep resident experts 4-bit; fetch synchronous misses at 2-bit:
target/release/cherenkov /path/to/model 'Explain hash collisions.' --miss-experts 2
# Opt into a disk-timing-dependent deadline policy:
target/release/cherenkov /path/to/model 'Explain hash collisions.' --cut-weak 0.08
```

Lower precision changes outputs. GPU and CPU arithmetic also differ in
4-bit mode. **A nonzero `--cut-weak` makes output non-reproducible**: a weak
expert is skipped when its read arrives late. The executable warns when
enabled. See the [saved baseline](../results/baseline-2026-09-09/README.md)
for measured settings and outputs; mixed-mode results using the deadline
cut must not be attributed to precision alone.
Routed-expert precision applies to batched prefill as well as decode.
Uniform 2/3-bit runs read the same compressed records in both phases.
In mixed mode, prefill preserves resident precision, fills its kept pool
entries at 4-bit, and streams transient misses at `--miss-experts` precision.
Shared experts and dense projections retain their existing precision.
This changes low-bit outputs relative to the saved baseline above, which
used 4-bit batched prefill; those baseline rates are historical measurements
of the earlier executable. The [paired prefill comparison](../results/prefill-lowbit-2026-09-09/README.md)
measures this change at 121, 521, and 3,221 prompt tokens. A separate
[mixed Q4/Q2 comparison](../results/prefill-mixed-2026-09-09/README.md) uses
the same binaries and prompt lengths, without a deadline cut.

Build one or both derived stores directly, without loading the GPU engine:

```sh
target/release/cherenkov pack /path/to/model --experts 3
target/release/cherenkov pack /path/to/model --experts 2,3
```

Omit the model path to use the managed download. `pack` defaults to 4-bit;
low-bit selections also create the aligned Q4 base if needed. Both missing
variants are generated in one pass through the base records. Valid stores
are reused. The first 2- or 3-bit inference run also builds a missing derived
store and announces it before starting. Construction time is reported
separately from inference.
Allow roughly 39 GB for 2-bit or 54 GB for 3-bit in addition to the base
store. `--repack` forces rebuilding the selected low-bit store. Multiple
stores can remain on disk, but only one is attached per run: mixed mode
requires `--experts 4`. Existing stores are not automatically deleted.

## Memory, context, and measurement

The adaptive pool leaves headroom for macOS; the default 2,048-token
configuration uses about **21 GB of Metal allocations** on the tested
M4 Air, within the roughly 25 GB target. This is not total process or
system memory: the OS also caches mapped files. `--pool-gb N` sets the
expert pool alone in decimal GB; `--pool-gb max` tries the device limit
and may put pressure on other applications.

Context takes about 22.5 KB per token from the expert pool. At 32,768,
131,072, and 262,144 tokens that costs roughly 4%, 17%, and 34% of the
usual pool. `--max-ctx` reserves capacity rather than setting an output
limit. The runner rejects a prompt + requested output + draft lookahead
that exceeds capacity before loading the GPU model.

`--max-tokens N --no-eos --repeat N` supports measurements with loaded
weights and fresh sequence state. Load, prefill, and decode are reported
separately on stderr; generated text goes to stdout. `--check` compares
rows with the CPU oracle and is deliberately slow; its timing is not an
inference benchmark.

Warm `--repeat` runs retain the expert pool. Resident-first accumulation
can change floating-point rounding and occasionally argmax as that pool
warms, even with the cut off; this also occurs in the original engine.
Use separate fresh processes with the same options for byte comparisons.
