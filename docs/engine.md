# Engine guide

Cherenkov has two execution paths for the same qwen4-exp model. The CPU
path is the slow numerical reference used by `--check`. The native Metal
path streams routed expert records through a bounded pool and runs the
rest of the network from resident weights. Neither uses MLX at runtime.

## Where to read

| Concern | Host code | Metal code |
| --- | --- | --- |
| Model configuration and packed layout | `src/qwen4_exp/config.rs`, `manifest.rs` | Host/shader layout contracts |
| Paths and checkpoint downloads | `src/storage.rs`, `src/download.rs` | None |
| Generation, draft acceptance, EOS | `src/runner.rs` | `common/sampling.metal`, `qwen4_exp/mtp.metal` |
| Shared GPU types, buffers, and state fields | `src/qwen4_exp/gpu.rs` | Both kernel libraries |
| Loading and pipeline creation | `src/qwen4_exp/gpu/load.rs` | Sources assembled by `src/kernels.rs` |
| Host/shader parameter layouts | `src/qwen4_exp/gpu/params.rs` | Matching subsystem structs |
| Trunk stepping and draft execution | `src/qwen4_exp/gpu/decode.rs`, `gpu/mtp.rs` | Subsystem kernels |
| Rollback and prefix checkpoints | `src/qwen4_exp/gpu/state.rs` | Recurrent snapshots and copying |
| Router/CPU handshake and lookahead reads | `src/qwen4_exp/gpu/streaming.rs` | `qwen4_exp/experts.metal` |
| Expert slots, residency, parallel file reads | `src/qwen4_exp/gpu/residency.rs` | Tagged record addresses consumed by `fn_moe_*` |
| Long-prompt orchestration | `src/qwen4_exp/gpu/prefill.rs` | Subsystem kernels |
| Prefill attention and expert ring | `src/qwen4_exp/gpu/prefill/attention.rs`, `prefill/experts.rs` | `common/gemm.metal`, `qwen4_exp/expert_gemm.metal`, `qwen4_exp/qsa.metal`, `qwen4_exp/rows.metal` |
| CPU equations and state | `src/qwen4_exp/cpu.rs` | DeltaNet, QSA, hyper-connection and PLE kernels |
| Derived 2/3-bit store layout | `src/qwen4_exp/lowbit.rs` | Q2/Q3 helpers in `qwen4_exp/experts.metal` |
| Metal allocation and mapped buffers | `src/metal.rs` | Shared-buffer access; probe in `device/clock.metal` |

Kernel paths above are relative to `kernels/`. Structs marked `repr(C)`
are a host/shader ABI: update the matching Metal struct whenever changing
fields, their order, or types. Offsets passed to `bind` are bytes; tensor
indices inside kernels are elements unless stated otherwise.

The [kernel map](../kernels/README.md) lists each file's responsibility.
`src/kernels.rs` assembles the common and qwen4-exp libraries for both the
engine and its GPU tests; fragments keep their own filenames in compiler
diagnostics. Splitting files changes organization, not the number of
libraries, command buffers, or kernel dispatches.

Decode dispatch lives in the `gpu/attention`, `deltanet`, `experts`,
`hyperconnection`, `ple`, and `sampling` child modules. Prefill has matching
subsystem children, plus allocation and projection helpers. Shared GPU
types remain in the parent; methods needed across children are visible
only within their parent subsystem. GPU tests mirror attention, QSA,
sampling, and state under `tests/unit/qwen4_exp/gpu/`.

## Decode and expert IO

A step processes one input token plus up to three MTP drafts. Buffers
are row-major `[nb][...]`. The hyper-connection residual contains four
hidden-width streams. A block can leave its MoE output pending: the next
fused normalization injects it into the residual before reading it. PLE
applies a pending output before its own transformation.

Each block has a four-stage handshake. Values are relative to that
block's `seq`; the resident-completion signal uses a separate event.

| Value | Writer | What has become safe |
| --- | --- | --- |
| `seq` | GPU | Router indices/weights are ready for the CPU to read. |
| `seq + 1` | CPU | The slot table and per-row weights are published; the GPU may read resident experts. |
| `seq + 2` on `event_res` | GPU | Resident expert computation finished; this is the optional deadline boundary. |
| `seq + 3` | CPU | Required misses have landed and joined residency; the GPU may read the remaining table entries. |

The table contains a first-seen union of routed experts, ordered with
residents first and then by descending maximum routing weight across
rows. The final two entries hold resident count and total active count.
The high bits of a record address identify its precision (bit 63 for
3-bit, bit 62 for 2-bit); the shader masks them before dereferencing.

### Block expert address tables

The router selects expert IDs and weights independently for each token row,
including speculative draft rows. The CPU forms their union and resolves
each `(layer, expert)` pair to a slot in the LRU pool. `slot_tab` stores
the slots' GPU addresses with precision tags; `wmap` stores a routing weight
for each row and union entry. An expert absent from a row has weight zero.
The final two table entries hold the resident and total expert counts.

For example, suppose the token routes to A/B and a draft routes to B/C.
If A and C are cached while B needs a read, the table is ordered A/C/B:

| Table entry | Address | Token weight | Draft weight | Ready |
| --- | --- | --- | --- | --- |
| A | A's current pool slot, tagged with its precision | A's weight | 0 | Resident |
| C | C's current pool slot, tagged with its precision | 0 | C's weight | Resident |
| B | B's reserved pool slot, tagged with its precision | B's weight | B's weight | After its read |

Metal kernels follow these addresses and use the tags to select Q4, Q3 or
Q2 dequantization. The CPU publishes the resident portion first; the GPU
computes it while missing records are read into their reserved slots.
The misses-ready signal releases the fetched portion. Reusing a cache slot changes
the table, not the network's stored weights. Slots used by the current step
are protected from eviction.

Keep this ordering when refactoring. The GPU accumulates the resident
and fetched parts separately, and floating-point rounding can change
when residency changes. A warm repeat is therefore not a substitute for
a fresh-process output comparison.

Lookahead predicts the next layer's routes. Its reads start only after
the current required misses have finished, keeping prefetch traffic off
the critical reads. With a deadline cut, weak unread experts may instead
be removed from the end of the active table. Their reads remain tracked
until completion; those slots cannot be reused early. Disk timing then
affects the answer.

## Prefill

Long prompts run layer by layer. Dense projections use matrix kernels.
Routed tokens are grouped by expert into row-index and routing-weight
arrays, then gathered, projected, and scattered back. The most-used
experts can remain in the pool for decode; the rest stream through a
64-record ring in groups of eight.

The GPU signals when it has finished consuming a group. The CPU waits
for that signal before overwriting reused ring slots, fills the next
group, and signals a separate event when the group is readable. Each
prefill event has one writer and monotonically increasing values.
Routed-expert prefill and decode share the record-layout description and
store files. Batched prefill selects Q4 or Q2/Q3 GEMMs per record; its
8/16/32-token tiles stage dequantized half weights and accumulate in float.
Decode retains its specialized half-dot kernels and reduction order.

Uniform low-bit runs fill the pool and streaming ring from that low-bit
store. Mixed mode preserves each resident slot's precision; new kept
records use the pool's default (4-bit), while transient ring misses use
`--miss-experts`. Records kept for decode need no second load. Ring slots
retain their original Q4-sized pitch so either format fits without changing
the allocation budget. The existing event handshake protects slot reuse.
Shared experts and dense projections remain Q4; deadline cuts remain in
the decode path. Prefill telemetry sums actual record bytes by precision.
The ring-wait counter measures CPU waiting for GPU consumption, not disk
read latency.

## Recurrent state and rollback

For one DeltaNet head, the reference stores `S[key_lane][value_lane]`.
After normalized q/k and the causal convolution, each token performs:

1. Decay the state: `S = decay * S`.
2. Form the correction: `delta = beta * (v - S^T k)`.
3. Update the state: `S = S + k delta^T`.
4. Read the output: `y = S^T q`, followed by gated RMS normalization.

The GPU stores the transpose, `[value_lane][key_lane]`. A simdgroup
owns one value lane and spreads its 128 key lanes across 32 threads,
four floats each. It keeps that row in registers across a small batch.
The math is the same, but reduction order and half-precision projections
mean the CPU and GPU need not produce identical logits.

Verification snapshots recurrent state and convolution history after
potentially accepted rows. `commit(n)` restores the plane after row
`n - 1` when the batch was only partially accepted, then advances the
logical position. Positional attention/PLE entries beyond that position
are excluded or overwritten. MTP has its own KV length and following-token
dependency; prefix-cache reuse validates that dependency too.

Do not casually reorder reductions, turn half accumulators into float,
or replace explicit unrolling with loops: these details affect both the
numerical baseline and GPU register pressure. The extra eight-row dense
projection path is used by MTP's expanded hyper-connection rows even
though decode verification itself is capped at four rows.

## Shared memory ownership

Mapped dense weights stay alive through the packed model borrowed by
`Gpu`. Wrapped expert regions stay owned by their pool/mapping.
Shared storage permits CPU and GPU access to the same pages; events and
completed command buffers determine when a particular region may be
read or overwritten. Sharing an address does not permit simultaneous
unsynchronized reads and writes.

The default expert pool is one shared Metal allocation kept resident for
the command queue. CPU reads fill its slots directly, using the CPU view
of the same physical memory the GPU addresses. Packed dense weights use
`newBufferWithBytesNoCopy` over their file mapping. The optional
`CHERENKOV_POOL=set` mode instead wraps individual file-backed expert regions;
it is not the default streaming path and supports Q4 only.

### N-gram offloading

The n-gram store is mapped on the CPU and is not wrapped as a giant Metal
buffer. Its row IDs depend on token history, so each decode step or prefill
chunk can start prefetching the required rows before the PLE blocks need
them. CPU threads read through the page cache, then `Packed::ngram_row`
dequantizes the selected embeddings into small shared buffers. PLE projections,
gating and convolution run on the GPU. The OS can reclaim cached file pages;
the whole embedding table does not need to be GPU-resident.
