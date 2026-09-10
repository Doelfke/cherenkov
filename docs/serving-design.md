# Serving design: sampling, conversations, and bounded memory

Research and implementation proposal, 2026-09-09. Code inspected at
`ab7929c`. This document describes proposed
behavior; it does not claim these features are implemented.

The initial TOML configuration, local control CLI, statistics and prefix
cache retention are now implemented; see [server-config.md](server-config.md)
for the supported interface. The larger session/sampling design below
remains a proposal and needs the checkpoint, admission and scheduling
contracts tightened before implementation.

## Recommendation

Keep Chat Completions as the primary compatibility interface. Add a
text-only Responses adapter for response IDs and conversation continuation.
Both should normalize into the same request, scheduler, sampler, and state
store. Keep the native Rust/Metal engine. HTTP JSON and SSE are sufficient
for concurrent text streams; a WebSocket connection is not a prerequisite.

Prioritize correct model templating and sampling, then memory accounting
and retained sessions, then interleaved generation. Start by testing two
active sequences, with four as a benchmark target. These are scheduling
targets, not measured throughput or memory guarantees.

## API conventions worth adopting

| Interface | Relevant conventions | Application here |
| --- | --- | --- |
| OpenAI Chat Completions | `messages`, `temperature`, `top_p`, `max_completion_tokens`, `reasoning_effort`; JSON or SSE | Preserve existing clients and explicit greedy settings. [Reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). |
| OpenAI Responses / Conversations | `previous_response_id` chains responses; `conversation` holds conversation items | Use these for retained history. They are distinct continuation mechanisms; do not combine them in one request. [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state), [Responses reference](https://platform.openai.com/docs/api-reference/responses-streaming). |
| OpenAI prompt caching | `prompt_cache_key` groups cache use; retention controls depend on the model | A cache key is not a conversation ID and does not replace history. Do not promise OpenAI's retention guarantees for a bounded local cache. [Caching guide](https://developers.openai.com/api/docs/guides/prompt-caching). |
| vLLM | OpenAI-compatible endpoints; `top_k` through SDK `extra_body`; request-ID headers | Accept common sampling extensions and normalize them once. `extra_body` merges into the JSON body; it is not a literal wire field. [Server docs](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/). |
| llama.cpp | Parallel slots, `id_slot`, prompt reuse, RAM cache limit, checkpoint limits | Borrow separate active-slot and cache budgets. Keep physical slot IDs internal so clients do not depend on placement. [Server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). |
| Ollama | `/api/chat`, `options`, `think`, `keep_alive`; `num_ctx` controls context | Useful naming precedents. `keep_alive` retains a model, not a conversation. Native streaming uses its own format; no need to add another adapter initially. [Chat](https://docs.ollama.com/api/chat), [FAQ](https://docs.ollama.com/faq). |
| Anthropic Messages | Stateless message history; explicit or automatic `cache_control` | Cache retention and conversation persistence are separate concerns here too. Add this adapter only for a client requirement. [Messages](https://platform.claude.com/docs/en/api/messages/create), [Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). |

There is no common portable request field for local resident-memory caps,
per-session byte limits, or eviction policies. Put operator settings in a
typed server configuration. Use a documented `cherenkov` request extension
only where a client needs a smaller context or shorter retention policy.
Reject unsupported behavioral options instead of silently ignoring them.

## The exact model template

The conversion card points to the preserved upstream card. That card
documents thinking enabled by default, effort levels `low`, `medium`, and
`xhigh` (default), and `preserve_thinking=true`. Recommended sampling is:

| Mode | Temperature | Top-p | Top-k | Presence penalty |
| --- | ---: | ---: | ---: | ---: |
| Thinking | 1.0 | 0.95 | 20 | 0.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 1.5 |

Both use min-p 0 and repetition penalty 1. These are upstream
recommendations, not measured Cherenkov settings. Lower effort need not
reduce total time across an entire agent task. [Upstream model card](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit/blob/6cc9bbc0fae9ce26b7670b3ed1e26d557c154506/README.upstream.md),
[conversion card](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit).

The local `chat_template.jinja` exactly matches the template embedded in
`tokenizer_config.json`, and the remote file at revision
`6cc9bbc0fae9ce26b7670b3ed1e26d557c154506`. Its SHA-256 is
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

The template inserts a system instruction for low or xhigh effort;
medium leaves that instruction empty. Thinking opens a reasoning block;
disabled thinking closes an empty block before generation. Assistant
history has specific reasoning-block formatting, whitespace trimming,
and preservation rules. [Exact template](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit/blob/6cc9bbc0fae9ce26b7670b3ed1e26d557c154506/chat_template.jinja).

Our CLI and server currently hand-build ChatML and force thinking off.
The server also does not preserve incoming `reasoning_content`. Implement
one model-template module shared by CLI and server, with byte/token
fixtures checked against the pinned template's text-only behavior. Either
evaluate that template with a compatible Rust renderer or faithfully
implement its supported text subset; test equivalence before choosing.
Template support alone does not implement vision or tool execution.

Normalize Chat `reasoning_effort` and Responses `reasoning.effort` to the
model's three levels. Support `chat_template_kwargs.enable_thinking` and
`preserve_thinking`. An explicit effort should enable thinking unless
contradicted by an explicit false flag, which should fail validation.
If offering `none`, document it as a Cherenkov alias for thinking off;
reject `high` rather than silently mapping it to `xhigh`.

Effort is model conditioning, not a kernel precision setting or an exact
token budget. Keep a separate total generation cap counting reasoning
and answer tokens. A future hard reasoning budget needs a specified
termination policy; injecting a closing tag can affect the answer and
must not be presented as the model's native effort behavior.

Parse generated reasoning into a separate channel, including tags split
across tokens/chunks. Preserve it in stored assistant history according to
the template policy. For Chat, follow the model card's `reasoning_content`
extension. Responses needs its own typed events; do not relabel raw model
reasoning as an OpenAI-generated reasoning summary.

Keep existing greedy/non-thinking defaults during implementation and make
all benchmark settings explicit. Switching defaults to the model card's
recommendations is a separate visible behavior change.

## Current engine constraints

| Component | Current behavior | Needed change |
| --- | --- | --- |
| `src/server.rs` | One generation runs to completion; queue holds eight jobs; up to 32 HTTP reader threads | Bounded scheduler and independent stream writers. Reader count is not a session limit. |
| `src/runner.rs` | Blocking loop with greedy MTP verification | Resumable prefill/decode state machine and per-request sampler. |
| `src/qwen4_exp/gpu.rs` | Weights, mutable layer state, scratch, MTP and profiling live in one object | Separate shared model/runtime from sequence state and request statistics. |
| `src/prefix_cache.rs` | Exact token-prefix lookup, byte LRU, 512 MiB default / 2 GiB maximum; snapshots during prefill | Session ownership, TTL/count/entry limits, committed completion checkpoints. |
| `src/qwen4_exp/gpu/state.rs` | Full hybrid snapshots copied into CPU vectors | Versioned compatible state; ownership-aware resident reuse and compact idle snapshots. |
| `src/metal.rs` | Subsequent `new_buffer` calls check Metal allocation plus reserved cache allowance | One budget covering all allocations from startup, including CPU state and temporary copies. |

The four rows supported by `MAX_NB` are consecutive positions in one
sequence, including speculative drafts. They are not four independent
sessions. Context capacity is currently allocated globally at load time.
Merely adding a request parameter would not reduce allocation size.
Nonzero temperature is rejected today; fields such as `top_k` and
reasoning controls are not validated and can be silently ignored. Close
that validation gap before advertising the expanded interface.

## Sampling and MTP

Implement validated temperature, top-k, top-p, seed, and presence/frequency
penalties together. Presence penalty is needed to reproduce the card's
non-thinking recommendation. Keep temperature zero on the existing greedy
fast path when other sampling controls are neutral. Specify sampler order
and penalty history scope; test against a small independent reference.

The target already produces full logits. There are 248,320 float logits
per row, about 0.95 MiB. A CPU sampler over synchronized shared memory is
a reasonable first implementation to measure. Reuse bounded workspace,
avoid copying all logits unnecessarily, and move selection to Metal if
profiling justifies it. Each request owns its RNG state; scheduler order
must not select its random numbers.

Current MTP accepts drafts by target argmax equality. Applying temperature
only after that verification would produce the wrong sampling behavior.
First establish a correct sampled baseline with speculation disabled for
non-greedy requests. Then test target-sample-and-match verification:
sample the target distribution at each valid proposed position; accept
the draft only if it matches, and keep the sampled target token on the
first mismatch. Discard later rows and roll back state. This works with
deterministic draft proposals, but acceptance can fall at higher
temperature. Consume random draws only along the retained prefix.

A more efficient stochastic-draft verifier can use acceptance/rejection
with target and proposal probabilities, including the residual
distribution. That requires additional draft distribution plumbing.
[Speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html).
Distribution tests should use fixed logits; this engine's changing expert
residency and batch arithmetic can independently change computed logits.

Exact-prompt cache hits currently reuse a cached next token and drafts.
Store reusable logits or the state needed to regenerate them, and sample
afresh with the current settings. Do not replay an earlier random result.
MTP's following-token dependency must still be validated or repaired.

## IDs and conversation lifetime

Use separate opaque, randomly generated identifiers for requests,
responses, and conversations. Physical sequence-slot indexes are internal.
Return a request ID in headers, use response IDs in JSON and every stream
event, and expose a conversation ID only for explicitly retained history.
An ID is not authentication and a request ID is not an idempotency key.

Chat requests remain stateless: `messages` is the complete intended
history. An optional session/cache affinity hint must not cause the server
to append that history a second time. For clients wanting incremental
turns, add the text-only Responses subset with `previous_response_id`,
then Conversations create/retrieve/delete and item storage. An immutable
response parent permits explicit forks. Serialize mutation of a mutable
conversation, or reject conflicting turns with a clear conflict error.

Store canonical messages, reasoning, tokenization/template identity,
response lineage, and committed token boundaries independently of GPU
checkpoints. Deduplicate retained parent history rather than copying the
whole transcript into every response. Count references and branches in
retention accounting; no unbounded ancestor retention.

Checkpoint eviction keeps the ID valid while its history survives: rebuild
state from the nearest compatible checkpoint or replay the transcript.
History expiry/eviction makes the ID unavailable and returns an explicit
not-found/expired-ID error. Never silently begin an empty conversation.
Expose effective expiry and state availability in the local session
metadata. Initial persistence is process-local; restart invalidates IDs.

Cache reuse requires matching actual rendered token prefixes, model/store
identity and numerical configuration. Effort changes system conditioning;
reasoning preservation and whitespace rules can rewrite earlier history.
Consequently, continuation cannot blindly assume the old state is an
append-only prefix. Keep sampling outcomes/RNG separate from reusable
forward state. Keep cache namespaces explicit when clients request them.

The generation loop currently commits accepted speculative rows before
checking EOS/output limits while emitting those rows. A final snapshot
must represent exactly the retained conversation boundary, with pending
next-token state represented explicitly. Resolve this before caching
generated answers, cancelling jobs, or suspending them.

## Memory ownership and limits

Apple Silicon's CPU and GPU share physical memory. A CPU copy of GPU
state consumes additional unified RAM; changing storage mode does not
create a second memory budget. [Apple storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).

Use one allocation ledger, with leases returned when the last owner and
GPU/IO use have finished. Reserve before allocating, including page
rounding, vector capacity, scratch growth and old-plus-new resize peaks.
Track shared backing allocations once even when several views reference
them. A cached copy has its own charge. Route `new_buffer`, no-copy
wrappers, mapped resident regions, CPU snapshots and request buffers
through the same accounting rules.

The managed budget must cover:

```text
shared dense/model allocations + resident expert pool
+ active and retained-idle sequence allocations
+ shared decode/rollback/prefill/IO scratch
+ compact checkpoint cache + retained transcript/response storage
+ bounded queued requests, stream buffers, and runtime reserve
<= configured memory budget (initial target: 25 decimal GB)
```

This is an enforceable allocation budget, not a claim that Metal's
`currentAllocatedSize` measures whole-process physical memory. Report
Metal allocation, managed bytes, process physical footprint/high-water,
and OS memory pressure separately. Mapped file size is not residency, and
reclaimable file cache is controlled by macOS. Preserve system headroom;
monitor faults/swap and footprint during sustained mixed workloads.

Suggested configuration groups (names proposed, not existing flags):

| Group | Controls |
| --- | --- |
| Total memory | `memory.max_bytes`, `memory.runtime_reserve_bytes` |
| Experts | `experts.resident_max_bytes`; optional adaptive pool within the remaining budget |
| Resident sequences | `resident.max_sequences`, `resident.max_bytes`, `resident.idle_ttl` |
| Scheduler | `scheduler.max_active`, `max_queued_requests`, `max_queued_bytes`, `max_queued_tokens`, `queue_timeout`, `max_stream_buffer_bytes` |
| Checkpoint cache | `cache.max_bytes`, `max_entries`, `max_entry_bytes`, `max_entries_per_session`, `idle_ttl`, `max_age`, `eviction = lru` |
| Retained sessions | `sessions.max_count`, `max_total_bytes`, `max_session_bytes`, `max_messages`, `max_responses`, `idle_ttl`, `max_age`, `eviction = lru` |
| Context/generation | `context.default_tokens`, `context.max_tokens`, per-request `context_tokens`, total output cap and generation timeout |

Resident sequence count includes active and retained-idle states;
`max_active <= resident.max_sequences`. Anonymous requests consume active
slots and bytes too. Session history and checkpoint limits apply to all
branches and include reasoning text, token buffers and metadata.
Per-request overrides can lower operator limits, never raise them.
Define zero consistently as disabled where meaningful; disallow zero
active slots and reject contradictory or overflowing configurations.

For an initial two-session experiment, try a 2 GiB resident-state ceiling,
a 1 GiB checkpoint ceiling, a 64 MiB transcript ceiling and an 8K default
context. Reserve shared scratch separately. These caps must fit alongside
the chosen expert pool inside 25 GB; they are not additive allowances on
top of the current pool. Choose pool capacity from the complete startup
plan. Reject an explicit infeasible plan rather than quietly exceeding it.
The existing copy pool is one large allocation, so reducing logical slot
count alone does not free memory. Dynamic pool shrink would require
segmented backing and should be a separate change.

Use bounded context buckets initially: allocate each sequence for its
admitted prompt plus maximum output and speculative lookahead, rounded
to a bucket, and reserve that capacity before starting. Honor a smaller
request cap; enforce the server and model maxima. Grow between turns only
after reserving the replacement allocation. Default overflow behavior is
an up-front error. Truncation would require rerendering/replaying hybrid
state; removing old KV entries alone cannot forget DeltaNet/PLE history.

Approximate compact checkpoint sizes with the current full model and MTP:

| Used tokens | Checkpoint size | Maximum fitting in 1 GiB, ignoring metadata |
| ---: | ---: | ---: |
| 2,048 | 156.5 MiB | 6 |
| 8,192 | 288.1 MiB | 3 |
| 32,768 | 814.8 MiB | 1 |

These are calculated from `prefix_regions`, not measured active-session
allocations: about 112.57 MiB fixed state plus 22,472 bytes per token,
with rounding and metadata. Active state also needs capacity rather than
used length. The existing per-layer rollback planes must move into shared
execution scratch to avoid multiplying them for every resident session.
Do not advertise a session count by dividing RAM by KV bytes alone.

## Retention, eviction, and pressure behavior

Maintain three lifetimes: ready-to-run resident sequence state, compact
replay checkpoints, and conversation history. Suggested initial idle TTLs
are 60 seconds, 15 minutes, and one hour respectively; use explicit
absolute-age caps too (for example one hour for checkpoints, 24 hours for
history). These are proposed local policies, not remote API guarantees.
TTL is eligibility for retention, not a promise against earlier pressure
eviction. Report the configured behavior clearly.

1. Expire idle entries and evict unpinned LRU checkpoints before new
   allocations. Oversized checkpoints are skipped; generating the answer
   should not fail merely because its checkpoint cannot be cached.
2. Release idle resident sequence allocations when their TTL/count/byte
   limit is reached. Compact them only if destination and temporary-copy
   budgets fit; otherwise keep the history and replay later. Transferring
   ownership of an existing buffer avoids a copy but still charges its
   full allocated capacity.
3. Bound each session's checkpoints so one long prefill cannot fill the
   cache with near-duplicate full snapshots. Share immutable checkpoint
   ownership where possible. Defer paged KV deduplication until needed;
   recurrent state still requires explicit boundary snapshots.
4. Pin active work and referenced state until GPU command buffers and
   expert reads finish. Cancellation removes scheduling eligibility first;
   it does not authorize premature buffer reuse. [Apple synchronization](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work).
5. When history capacity is exhausted, evict eligible idle history under
   the documented policy. If every candidate is pinned, reject admission.
   If a single conversation exceeds its own limit, reject its new turn
   with an actionable size error; do not silently discard older messages.
6. Reserve capacity for the retained answer before admitting a stored
   response, using bounded token/text expansion and metadata estimates.
   Check growth while streaming. Every queue, output buffer, response
   index, expiry index and profiling collection must also remain bounded.
7. On pressure, reclaim idle caches and lower admissions first. If active
   reservations cannot fit, queue within limits or return 503 with a retry
   hint. Avoid repeatedly swapping the same active sessions every token.

Disk-backed state is optional later, with its own byte cap and expiry.
Start with it disabled; SSD writes and page cache still have costs. Do not
reuse the expert packing directory as an unbounded session dump store.

## Concurrent generation without duplicating the engine

Split the runtime into shared weights/pipelines/expert pool, per-sequence
KV/QSA/DeltaNet/PLE/MTP state, per-request generation/RNG/decoder state,
and shared execution scratch. Keep the Metal engine on one owner thread.

Turn the runner into `prefill_chunk` and `decode_step` operations that
return at a committed boundary. Fairly interleave active sequences at
those boundaries. Bound prefill chunks so a long new prompt cannot block
existing streams for its entire prefill. The first implementation executes
one GPU step at a time while multiple clients make progress.

Avoid snapshotting/restoring hundreds of MiB on each switch. Bind each
resident sequence's buffers directly and share scratch only after its
previous user has finished. Global event counters remain monotonic;
request state must not restore them. Keep expert slots pinned through
their last GPU/IO use. Finish a speculative step and commit before yielding.

Move TCP writes off the engine thread into bounded per-response channels.
Handle disconnects, backpressure and timeouts explicitly; one slow client
must not stall all GPU work. A mature HTTP/SSE layer may simplify this,
but a networking rewrite is not required to design the engine boundary.

Cross-sequence microbatching is a later kernel change: sequence IDs,
per-sequence positions/lengths and state pointers must reach attention,
DeltaNet, PLE and MTP. MoE can then union experts across sequences, but
working sets may grow and reduce residency. Benchmark aggregate throughput
and per-client latency; concurrency alone does not imply a speedup.

vLLM's separate limits for scheduled sequences, batched tokens, queued
requests and queued prompt tokens support the same separation of concerns.
[Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/).

## Implementation sequence and acceptance checks

| Phase | Main files / work | Completion evidence |
| --- | --- | --- |
| 1. Template and sampler | Shared template module; typed request options; `runner.rs`, `gpu/sampling.rs`, response reasoning channel | Pinned-template fixtures for all efforts, thinking off, multi-turn preservation; sampler distribution/seed tests; unsupported controls rejected. |
| 2. Memory foundation | Allocation leases and startup planner in `metal.rs` / GPU load; split weights, sequence state and scratch | Count every backing allocation once; failure/resize/cancellation paths release reservations; peak fits configured budget. |
| 3. IDs and retained state | Session store, bounded TTL/LRU cache, final-state checkpoints, Responses adapter | Resume/fork, expiry/delete, oversized entry, strict limits, changed settings and EOS/speculative-boundary tests. |
| 4. Concurrent progress | Resumable runner, scheduler, bounded stream writers | Two interleaved clients progress; long prefill and slow/disconnected clients do not starve others; bounded queue and memory. |
| 5. Performance | Sampled MTP verification, then optional cross-sequence batching | Distribution correctness and measured benefit; no unmeasured speed claim. |

Keep tests in separate files under the existing test hierarchy. Use fake
clocks/allocators for retention and admission tests; real Metal tests for
state isolation, synchronization and capacity changes. Check byte-identical
state restoration; separately characterize numerical differences from
prefill chunking, batching and resident expert ordering.

Benchmark active counts 1/2/4, short and 8K/32K contexts, cold/reused/evicted
prefixes, greedy and sampled output, and all effort modes. Include one
long-prefill client alongside decoders, abandoned streams, repeated turns,
and a churn run exceeding cache/session limits. Record aggregate and
per-client decode rate, uncached prefill rate, TTFT/queue latency, maximum
inter-token gap, cache hits, evictions/replays, expert misses, all memory
high-water marks, faults/swap and power state. Separate model load, packing,
queueing, cache restore, prefill and decode timings. Preserve complete
outputs, including pelicans, in the benchmark suite.
