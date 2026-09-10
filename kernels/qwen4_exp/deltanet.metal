// DeltaNet output normalization and sigmoid gating.

// The host binds DeltaPrepParams here too; keep the full layout identical
// to that Rust struct and the DeltaPrepParams in common/deltanet.metal.
struct GateNormParams {
    uint n_k;
    uint n_v;
    uint d_k;
    uint d_v;
    float eps;
    uint nb;
    uint snap_after;
};

// DeltaNet output: (norm(y) * w) * sigmoid(z), per (row, head).
kernel void fn_delta_gate_norm_sigmoid_b(
    device float*        y  [[buffer(0)]],
    device const float*  z  [[buffer(1)]],
    device const bfloat* nw [[buffer(2)]],
    constant GateNormParams& p [[buffer(3)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint iv   [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float shm[32];
    const uint dv = p.d_v;
    const uint b = tgpos / p.n_v;
    const uint head = tgpos % p.n_v;
    const ulong off = (ulong)b * p.n_v * dv + (ulong)head * dv + iv;
    float yv = y[off];
    float ss = simd_sum(yv * yv);
    if (lane == 0) shm[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < (dv + 31) / 32; i++) tot += shm[i];
    float inv = rsqrt(tot / (float)dv + p.eps);
    float zv = z[off];
    y[off] = (yv * inv * (float)nw[iv]) * (1.0f / (1.0f + exp(-zv)));
}
