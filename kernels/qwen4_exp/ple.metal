// PLE n-gram gating and dilated convolution.

// ---- PLE (n-gram) block, batched ----

// gated[b][g*n+i] = sigmoid(sq(dot_bg)) * value[b][i] with
// dot_bg = key[b][g] . query[b][g] / sqrt(n), sq(x) = sign(x) sqrt(max(|x|,1e-6)).
// One threadgroup per (row, group).
kernel void fn_ple_gate_b(
    device const float* key   [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float*       gated [[buffer(3)]],
    constant GroupParams& p   [[buffer(4)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    const uint b = tg / p.groups;
    const ulong base = (ulong)tg * p.n;
    float acc = 0.0f;
    for (uint i = tid; i < p.n; i += tpg) acc += key[base + i] * query[base + i];
    acc = simd_sum(acc);
    if (lane == 0) red[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint s = 0; s < (tpg + 31) / 32; s++) dot += red[s];
    dot /= sqrt((float)p.n);
    float gate = sqrt(max(fabs(dot), 1e-6f)) * (dot < 0.0f ? -1.0f : 1.0f);
    const float s = 1.0f / (1.0f + exp(-gate));
    device const float* vb = value + (ulong)b * p.n;
    for (uint i = tid; i < p.n; i += tpg) gated[base + i] = s * vb[i];
}

struct PleConvParams {
    uint channels;   // hc_hidden
    uint ksize;      // 4
    uint dilation;   // 3
    uint span;       // (kernel-1)*dilation history slots
    uint filled;     // positions already recorded (= position of row 0)
    uint nb;
};

// Dilated causal depthwise conv over time with SiLU, added to `gated`,
// for nb consecutive positions; each channel walks its rows in order and
// records gvn into the ring, so in-batch taps read what earlier rows wrote.
kernel void fn_ple_conv_b(
    device const float*  gated [[buffer(0)]],
    device const float*  gvn   [[buffer(1)]],
    device const bfloat* w     [[buffer(2)]],
    device float*        hist  [[buffer(3)]],
    device float*        out   [[buffer(4)]],
    constant PleConvParams& p  [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.channels) return;
    device const bfloat* wr = w + (ulong)c * p.ksize;
    for (uint b = 0; b < p.nb; b++) {
        const uint pos = p.filled + b;
        const ulong bc = (ulong)b * p.channels + c;
        float acc = (float)wr[p.ksize - 1] * gvn[bc];
        for (uint k = 0; k + 1 < p.ksize; k++) {
            const uint back = p.dilation * (p.ksize - 1 - k);
            if (back <= pos) {
                const uint slot = (pos - back) % p.span;
                acc += (float)wr[k] * hist[(ulong)slot * p.channels + c];
            }
        }
        out[bc] = gated[bc] + acc / (1.0f + exp(-acc));
        hist[(ulong)(pos % p.span) * p.channels + c] = gvn[bc];
    }
}
