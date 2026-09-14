// Hyper-connection replication, normalization, bottleneck projection, mixing, and injection.

// hyper[b][g][i] = e[b][i]
kernel void fn_replicate_b(
    device const float* e     [[buffer(0)]],
    device float*       hyper [[buffer(1)]],
    constant GroupParams& p   [[buffer(2)]],
    constant uint&      nb    [[buffer(3)]],
    uint gi [[thread_position_in_grid]])
{
    const uint hh = p.n * p.groups;
    if (gi >= nb * hh) return;
    const uint b = gi / hh;
    const uint i = (gi % hh) % p.n;
    hyper[gi] = e[b * p.n + i];
}

// One threadgroup per (row, group):
//   y[b][g*n+i] = rmsnorm(x[b][g*n..])[i] * (w[g*n+i] + shift)
kernel void fn_group_norm_b(
    device const float*  x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device float*        y [[buffer(2)]],
    constant GroupParams& p [[buffer(3)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    const uint g = tg % p.groups;
    const ulong base = (ulong)tg * p.n;
    const ulong wbase = (ulong)g * p.n;
    float ss = 0.0f;
    for (uint i = tid; i < p.n; i += tpg) {
        float v = x[base + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) partial[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float total = (lane < (tpg + 31) / 32) ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) partial[0] = rsqrt(total / (float)p.n + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint i = tid; i < p.n; i += tpg) {
        y[base + i] = x[base + i] * inv * ((float)w[wbase + i] + p.shift);
    }
}

struct NormPrepParams {
    uint n;        // group width (hidden)
    uint groups;   // streams
    float eps;
    uint inject;   // 1: first add out[b][i] * 2 sigmoid(r[b][g]/groups) into x
};

// Fused gated-residual read, stage 1, one 256-thread threadgroup per
// (row, group): optional injection of `out` into x (in place), grouped
// RMSNorm to y, and y's half even/odd copy plus per-32 group sums for the
// Q4 matvecs (deinterleave_bh layout). Requires n % 256 == 0.
kernel void fn_norm_prep_b(
    device float*        x    [[buffer(0)]],   // [nb][hh]
    device const bfloat* w    [[buffer(1)]],   // [hh]
    device float*        y    [[buffer(2)]],   // [nb][hh]
    device half*         xe   [[buffer(3)]],   // [nb][hh/2]
    device half*         xo   [[buffer(4)]],
    device float*        xsum [[buffer(5)]],   // [nb][hh/32]
    device const float*  out  [[buffer(6)]],   // [nb][n]
    device const float*  r    [[buffer(7)]],   // [nb][groups]
    constant NormPrepParams& p [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    const uint b = tg / p.groups;
    const uint g = tg % p.groups;
    const uint hh = p.n * p.groups;
    const ulong base = (ulong)tg * p.n;
    const ulong wbase = (ulong)g * p.n;
    float wgt = 0.0f;
    if (p.inject != 0) wgt = 2.0f / (1.0f + exp(-r[b * p.groups + g] / (float)p.groups));
    float ss = 0.0f;
    for (uint i = tid; i < p.n; i += tpg) {
        float v = x[base + i];
        if (p.inject != 0) {
            v += out[(ulong)b * p.n + i] * wgt;
            x[base + i] = v;
        }
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) partial[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float total = (lane < (tpg + 31) / 32) ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) partial[0] = rsqrt(total / (float)p.n + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    // With 256 threads, element i = tid + 256k sits in 32-chunk i/32 =
    // sgid + 8k with lane i % 32, so a simd_sum is the chunk sum.
    device float* xs = xsum + (ulong)b * (hh / 32) + wbase / 32;
    for (uint i = tid; i < p.n; i += tpg) {
        const ulong j = base + i;
        float v = x[j] * inv * (float)w[wbase + i];
        y[j] = v;
        if ((j & 1) == 0) xe[j >> 1] = (half)v; else xo[j >> 1] = (half)v;
        float s = simd_sum(v);
        if (lane == 0) xs[i >> 5] = s;
    }
}

// Q4 matvec over NB rows whose outputs go through silu(y / div): the
// gated-residual bottleneck. y is [NB][out_dim] f32.
#define FN_QMV_SILU_B_ARGS                              \
    device const uint*   w      [[buffer(0)]],          \
    device const bfloat* scales [[buffer(1)]],          \
    device const bfloat* biases [[buffer(2)]],          \
    device const half*   xe     [[buffer(3)]],          \
    device const half*   xo     [[buffer(4)]],          \
    device const float*  xsum   [[buffer(5)]],          \
    device float*        y      [[buffer(6)]],          \
    constant FnQmvParams& p     [[buffer(7)]],          \
    constant float&      div    [[buffer(8)]],          \
    uint tgpos [[threadgroup_position_in_grid]],        \
    uint sgid  [[simdgroup_index_in_threadgroup]],      \
    uint spt   [[simdgroups_per_threadgroup]],          \
    uint lane  [[thread_index_in_simdgroup]]

template <uint NB>
[[kernel]] void fn_qmv_silu_b(FN_QMV_SILU_B_ARGS)
{
    const uint row = tgpos * spt + sgid;
    if (row >= p.out_dim) return;
    const uint halves = p.in_dim / 32;
    device const uint4* wr = (device const uint4*)w + (ulong)row * halves;
    device const bfloat* sr = scales + (ulong)row * (halves / 2);
    device const bfloat* br = biases + (ulong)row * (halves / 2);
    float acc[NB];
    for (uint r = 0; r < NB; r++) acc[r] = 0.0f;
    fn_q4_rows_h<NB>(wr, sr, br, (device const half4*)xe, (device const half4*)xo, xsum,
                     halves, p.in_dim / 8, halves, lane, acc);
    if (lane == 0) {
        for (uint r = 0; r < NB; r++) {
            float v = acc[r] / div;
            y[(ulong)r * p.out_dim + row] = v / (1.0f + exp(-v));
        }
    }
}
#define FN_INST_QMV_SILU(N) \
    template [[host_name("fn_qmv_silu_b" #N)]] [[kernel]] void fn_qmv_silu_b<N>(FN_QMV_SILU_B_ARGS);
FN_INST_QMV_SILU(1)
FN_INST_QMV_SILU(2)
FN_INST_QMV_SILU(3)
FN_INST_QMV_SILU(4)
FN_INST_QMV_SILU(5)
FN_INST_QMV_SILU(6)

// mixed[b][i] = mean over groups of sigmoid(u[b][g*n+i]) * normed[b][g*n+i]
kernel void fn_hc_mix_b(
    device const float* u      [[buffer(0)]],
    device const float* normed [[buffer(1)]],
    device float*       mixed  [[buffer(2)]],
    constant GroupParams& p    [[buffer(3)]],
    constant uint&      nb     [[buffer(4)]],
    uint gi [[thread_position_in_grid]])
{
    if (gi >= nb * p.n) return;
    const uint b = gi / p.n;
    const uint i = gi % p.n;
    const ulong base = (ulong)b * p.n * p.groups + i;
    float acc = 0.0f;
    for (uint g = 0; g < p.groups; g++) {
        const ulong idx = base + (ulong)g * p.n;
        acc += normed[idx] / (1.0f + exp(-u[idx]));
    }
    mixed[gi] = acc / (float)p.groups;
}

// hyper[b][g][i] += out[b][i] * 2 sigmoid(r[b][g] / groups)
kernel void fn_inject_b(
    device float*       hyper [[buffer(0)]],
    device const float* out   [[buffer(1)]],
    device const float* r     [[buffer(2)]],
    constant GroupParams& p   [[buffer(3)]],
    constant uint&      nb    [[buffer(4)]],
    uint gi [[thread_position_in_grid]])
{
    const uint hh = p.n * p.groups;
    if (gi >= nb * hh) return;
    const uint b = gi / hh;
    const uint rem = gi % hh;
    const uint g = rem / p.n;
    const uint i = rem % p.n;
    const float wgt = 2.0f / (1.0f + exp(-r[b * p.groups + g] / (float)p.groups));
    hyper[gi] += out[(ulong)b * p.n + i] * wgt;
}
