# Serving architecture

The implemented interface is documented in [running](running.md) and
[server configuration](server-config.md). This note describes ownership and
the boundaries that keep concurrent requests independent.

## Source map

`src/server.rs` starts the server and connects its owners. Its child modules
separate the protocol from request execution:

| Module | Responsibility |
| --- | --- |
| `http` | Bounded HTTP input and JSON/SSE framing |
| `routes` | Endpoints, request registration and queue admission |
| `request` | Completion options, resolved session input and token/context validation |
| `registry` | Request IDs and the atomic cancellation/publication boundary |
| `worker` | GPU scheduling, checkpoint switching and active-memory admission |
| `sessions` | Retained history, sampling state and transactional turns |
| `output` | Bounded writer queues and final publication |
| `response` | Chat/completion response formats |
| `failure` | HTTP status classification |

`src/prompt.rs` loads and compiles the checkpoint's Jinja template once for
CLI/chat formatting. Cache boundaries are prefixes verified against the full
rendered text. Session settings and messages reach the worker as typed input;
resolving a session never rewrites the incoming request JSON.

## Ownership

| Owner | State |
| --- | --- |
| GPU worker | Shared weights, expert pool, pipelines, scratch and one live sequence |
| Active request | Hybrid checkpoint, prefill progress, pending token, verifier/drafts, sampler, decoder and output counter |
| Retained session | Committed chat history, sampling settings, RNG state and context cap |
| Prefix cache | Reusable hybrid checkpoints and final prompt logits, bounded by bytes/count/TTL |
| Response writer | Bounded text queue, socket and final-publication decision |

The worker schedules requests round-robin. It runs one prompt chunk or one
complete decode/verification step before yielding. It copies sequence state
into reusable CPU vectors when switching owners, then restores the next
sequence. With only one active request, it keeps the sequence on the GPU.
Global synchronization events and expert residency are shared and never
restored from a request checkpoint. This is interleaving, not cross-request
GPU batching; concurrent service does not imply higher aggregate throughput.

Checkpoints contain attention KV, QSA indices and compressed blocks, DeltaNet
recurrent state, convolution/PLE history, tokens, position and MTP state.
Scratch can be reused only after the previous GPU work completes. Active
checkpoint reservations include vector growth and sampler workspace; admission
waits when requests cannot fit together. See the configuration reference for
what the memory cap covers and which allocations have separate bounds.

## Sampling and speculation

Each active request owns its sampler and mutable RNG. Selection applies prompt
and output occurrence penalties, temperature, top-k and nucleus filtering.
Unmodified temperature-zero generation retains the GPU argmax fast path and
adaptive MTP verification. Sampled or penalized generation uses target logits
with MTP verification disabled, avoiding the biased distribution that would
result from filtering sampled output through greedy draft acceptance.

The pending next token is part of the decoder. Suspending a request never
resamples it, and another request cannot consume its random draws. At an
output-length boundary, the decoder does not draw an unused next token.
A session continues its committed RNG before sampling the first token of a
new turn. An explicit request seed restarts it; greedy turns consume no draws.
Penalty counts are rebuilt from the new turn's complete rendered prompt.

Exact-prompt prefix hits retain logits so each sampled request makes its own
selection. Cache compatibility also distinguishes drafting state and checks
MTP's following-token dependency. Prefix reuse does not retain another
request's sampled next token or RNG.

## Cancellation and publication

Request registration pins an opaque ID through queueing, execution and output.
A cancellation flag can be set without entering the GPU worker. The worker
checks it before admission, before work and after prefill chunks. In-flight
GPU dispatches finish before sequence state is reused.

Session turns are transactions. They pin prior history/settings/RNG, render
new messages and generate into provisional output. Preparing a successful
turn reserves history capacity. The writer atomically chooses completion or
cancellation immediately before the final response; only completion publishes
the prepared history and RNG. Dropping a failed/cancelled turn releases its
pin and reservation without altering the committed session.

Thus partial streamed text can be discarded after cancellation. Retrying
starts from the last completed turn, not from the cancelled partial output.
A disconnect after the completion decision cannot undo an already committed
turn. Writer queues and socket timeouts bound slow clients independently of
GPU scheduling; write failures and queue overflow cancel unfinished requests.

## Retention and context

Sessions and prefix checkpoints have independent byte, count and idle limits.
Idle sessions can be evicted; in-flight turns are pinned. A retained history
can survive prefix-cache eviction and be prefetched again on its next turn.
Completed output is retained as messages, not as a separate GPU checkpoint;
the next turn restores any matching prompt prefix and prefills the remainder.
Neither store spills to disk or survives server restart.

The server allocates GPU context buffers once for its maximum context. A
smaller session/request context limits admission and checkpoint estimates;
it does not shrink those GPU buffers. Binding separate resident sequence
buffers could avoid checkpoint copies but remains future work.

## Remaining work

Effort/thinking controls and the Responses API remain unsupported and their
request controls are rejected. The checkpoint template now supplies formatting,
with `enable_thinking=false` and its own reasoning-history defaults. Exposing
the template's `low`, `medium` and `xhigh` effort settings is separate work.
Independent Transformers fixtures cover template rendering and tokenization;
see [the reference inputs](../tests/fixtures/prompt/README.md).

Sampled MTP would require distribution-correct target-sample-and-match or
proposal/target rejection sampling, with draws consumed only along retained
positions. Cross-request batching would require sequence-aware kernels.
Neither optimization is part of the current scheduler.

## Validation

Tests cover distribution/seed behavior, pending-token suspension, session RNG
continuation and reseeding, cancellation at final publication, history-memory
failure, expiry and pinning. Metal checks restore all hybrid state byte for
byte and resume sampling after another request's greedy MTP step. Live smoke
checks exercise interleaved streams, prefill/queued cancellation, unchanged
cancelled history, successful continuation and released reservations.

These are correctness checks. They do not establish throughput at different
concurrency levels or guarantee identical text when expert accumulation order
changes. See [validation](validation.md) for commands and known limitations.
