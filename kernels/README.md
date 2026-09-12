# Metal kernels

`common/` contains shared primitives; `qwen4_exp/` contains model kernels.
`device/clock.metal` is the dependent-FMA throttling probe. Rust files contain
no embedded Metal kernels.

| Common file | Responsibility |
| --- | --- |
| `quantized.metal` | Affine-Q4 prefill GEMMs, verify matvecs, half-stream staging |
| `gemm.metal` | Half-input dense GEMM for attention |
| `attention.metal` | QK norm/RoPE, q8 KV cache, prefill staging, decode partials/combine |
| `deltanet.metal` | Causal convolution, gate preparation, recurrent scan and snapshots |
| `elementwise.metal` | SiLU multiply, residual addition, copying |
| `sampling.metal` | Token embeddings and greedy argmax |

| qwen4-exp file | Responsibility |
| --- | --- |
| `types.metal` | Projection/group ABI types used across subsystems |
| `quantized_rows.metal` | Q4 row helpers used by hyper-connections and experts |
| `hyperconnection.metal` | Replication, norms, bottleneck projection, mixing, injection |
| `experts.metal` | Router/top-k, address tables, Q2/Q3 helpers, gate/up/down/combine |
| `expert_gemm.metal` | Q2/Q3 routed-expert prefill GEMMs, specialized for 8/16/32 token tiles |
| `ple.metal` | N-gram gating and dilated convolution |
| `mtp.metal` | Draft-head input folding |
| `deltanet.metal` | Output normalization and sigmoid gating |
| `qsa.metal` | Block indexing/selection and selected prefill/decode attention |
| `rows.metal` | Zeroing, narrow projections, activation, gather/scatter, shared-expert addition |

## Compilation and dependencies

[`src/kernels.rs`](../src/kernels.rs) assembles two libraries and the probe
for both the engine and tests. Fragments cannot compile independently.
The assembler supplies `metal_stdlib`, the namespace, and `#line` directives.

Shared types precede their users; Q4 row helpers precede hyper-connections
and experts. Q2/Q3 and address-table helpers are private to `experts.metal`.
Helpers do not cross library boundaries.

Keep macros with their definitions and instantiations. Preserve the matching
lane and reduction order of the Q4 row helpers. Changes to projection
precision or batching require numerical and performance checks.

## Editor diagnostics

`mise run clangd` generates `kernels/.clangd`. It supplies the editor prelude
and force-includes preceding fragments. `check-metal` rejects stale config.
clangd parses MSL as C++; use the `check-metal` VS Code task for compiler errors.

Do not pass `--compile-commands-dir` to the Metal language server. It overrides
the generated config and can break relative includes.
