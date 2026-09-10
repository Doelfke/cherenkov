// DeltaNet causal convolution, gate preparation, and recurrent state scan.

struct ConvParams {
    uint channels;   // 10240
    uint ksize;      // 4
};

// Batched conv step: each thread owns a channel, iterating nb tokens.
kernel void conv_b(
    device float*        qkv  [[buffer(0)]],  // [b][channels]
    device const bfloat* w    [[buffer(1)]],
    device float*        hist [[buffer(2)]],
    constant ConvParams& p    [[buffer(3)]],
    constant uint&       nb   [[buffer(4)]],
    constant uint&       snap_after [[buffer(5)]],
    device float*        mid  [[buffer(6)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.channels) return;
    const uint k = p.ksize;
    const uint km1 = k - 1;
    device const bfloat* wr = w + (ulong)c * k;
    device float* h = hist + (ulong)c * km1;
    float h0 = h[0];
    float h1 = h[1];
    float h2 = h[2];
    float w0 = (float)wr[0];
    float w1 = (float)wr[1];
    float w2 = (float)wr[2];
    float w3 = (float)wr[3];
    for (uint b = 0; b < nb; b++) {
        device float* xc = qkv + (ulong)b * p.channels + c;
        float cur = *xc;
        float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
        h0 = h1; h1 = h2; h2 = cur;
        *xc = acc / (1.0f + exp(-acc));
        if (b < snap_after) {
            device float* plane = mid + (ulong)b * p.channels * km1;
            plane[(ulong)c * km1 + 0] = h0;
            plane[(ulong)c * km1 + 1] = h1;
            plane[(ulong)c * km1 + 2] = h2;
        }
    }
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
}

// DeltaNet recurrent scan; state is transposed for contiguous simdgroup loads.

struct DeltaPrepParams {
    uint n_k;      // 16
    uint n_v;      // 48
    uint d_k;      // 128
    uint d_v;      // 128
    float eps;
    uint nb;
    uint snap_after;  // save planes after rows [0, snap_after); 0 disables snapshots
};

// Per (token, k-head): L2-normalize k and q (query gets the 1/sqrt(dk)
// scale) out of the post-conv qkv into kqn = [b][ k(2048) | q(2048) ].
kernel void delta_norms(
    device const float* qkv [[buffer(0)]],
    device float*       kqn [[buffer(1)]],
    constant DeltaPrepParams& p [[buffer(2)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint spt   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint gsg = tgpos * spt + sgid;
    const uint total = p.nb * p.n_k;
    if (gsg >= total) return;
    const uint b = gsg / p.n_k;
    const uint kh = gsg % p.n_k;
    const uint dk = p.d_k;
    const uint qk = p.n_k * dk;
    const uint conv_dim = 2 * qk + p.n_v * p.d_v;
    device const float* qh = qkv + (ulong)b * conv_dim + kh * dk;
    device const float* kh_p = qkv + (ulong)b * conv_dim + qk + kh * dk;
    float4 q4 = *(device const float4*)(qh + lane * 4);
    float4 k4 = *(device const float4*)(kh_p + lane * 4);
    float qss = simd_sum(dot(q4, q4));
    float kss = simd_sum(dot(k4, k4));
    float qinv = rsqrt(qss + p.eps) * rsqrt((float)dk);
    float kinv = rsqrt(kss + p.eps);
    device float* out = kqn + (ulong)b * 2 * qk;
    *(device float4*)(out + kh * dk + lane * 4) = k4 * kinv;
    *(device float4*)(out + qk + kh * dk + lane * 4) = q4 * qinv;
}

// Per (token, v-head): decay and beta scalars into gb = [b][n_v][2].
kernel void delta_gates(
    device const float*  a     [[buffer(0)]],
    device const float*  bb    [[buffer(1)]],
    device const bfloat* a_log [[buffer(2)]],
    device const bfloat* dt_b  [[buffer(3)]],
    device float*        gb    [[buffer(4)]],
    constant DeltaPrepParams& p [[buffer(5)]],
    uint gi [[thread_position_in_grid]])
{
    if (gi >= p.nb * p.n_v) return;
    const uint h = gi % p.n_v;
    float av = a[gi] + (float)dt_b[h];
    float sp = av > 20.0f ? av : log(1.0f + exp(av));
    gb[2 * gi] = exp(-exp((float)a_log[h]) * sp);
    gb[2 * gi + 1] = 1.0f / (1.0f + exp(-bb[gi]));
}

// The scan. One simdgroup per (v-head, dv-row); state row (dk=128 floats,
// TRANSPOSED layout [head][dv][dk]) lives in 4 registers per lane across
// the whole scan. Two simd_sums per token; no threadgroup memory, no
// barriers. Emits raw readout y; fn_delta_gate_norm_sigmoid_b applies
// normalization and the output gate afterwards.
kernel void delta_scan2(
    device const float*  qkv   [[buffer(0)]],  // [b][conv_dim] post-conv
    device const float*  kqn   [[buffer(1)]],  // [b][k|q normalized]
    device const float*  gb    [[buffer(2)]],  // [b][n_v][decay, beta]
    device float*        state [[buffer(3)]],  // [head][d_v][d_k]
    device float*        y     [[buffer(4)]],  // [b][n_v*d_v] raw
    device float*        mid   [[buffer(5)]],  // snapshot plane, same layout
    constant DeltaPrepParams& p [[buffer(6)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint spt   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint gsg = tgpos * spt + sgid;
    if (gsg >= p.n_v * p.d_v) return;
    const uint head = gsg / p.d_v;
    const uint iv = gsg % p.d_v;
    const uint dk = p.d_k;
    const uint qk = p.n_k * dk;
    const uint conv_dim = 2 * qk + p.n_v * p.d_v;
    const uint hk = head / (p.n_v / p.n_k);

    device float* srow = state + ((ulong)head * p.d_v + iv) * dk;
    float4 s4 = *(device const float4*)(srow + lane * 4);

    for (uint b = 0; b < p.nb; b++) {
        device const float* kq = kqn + (ulong)b * 2 * qk;
        float4 k4 = *(device const float4*)(kq + hk * dk + lane * 4);
        float4 q4 = *(device const float4*)(kq + qk + hk * dk + lane * 4);
        float decay = gb[2 * (b * p.n_v + head)];
        float beta = gb[2 * (b * p.n_v + head) + 1];
        float vv = qkv[(ulong)b * conv_dim + 2 * qk + head * p.d_v + iv];

        // Decay the state, correct its key readout, then read with q.
        // Keep the float4/FMA/reduction order: it defines verify numerics.
        s4 *= decay;
        float kv_mem = simd_sum(dot(s4, k4));
        float delta = (vv - kv_mem) * beta;
        s4 = fma(k4, delta, s4);
        float yv = simd_sum(dot(s4, q4));
        if (lane == 0) {
            y[(ulong)b * p.n_v * p.d_v + (ulong)head * p.d_v + iv] = yv;
        }
        // Plane b is the state AFTER row b, restored by commit(b + 1)
        // when the following speculative rows are rejected.
        if (b < p.snap_after) {
            device float* plane = mid + (ulong)b * p.n_v * p.d_v * dk;
            *(device float4*)(plane + ((ulong)head * p.d_v + iv) * dk + lane * 4) = s4;
        }
    }
    *(device float4*)(srow + lane * 4) = s4;
}
