// General row utilities and narrow projections used by batched prefill.

kernel void fn_zero(
    device float*  x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) x[i] = 0.0f;
}

// ---- prefill engine helpers (row batches of any size) ----

// Q4 projection with few output rows over any number of input rows,
// straight from f32 x: one simdgroup per (row, output). For the narrow
// weights (injection logits, DeltaNet gates) the GEMM tiles waste or
// corrupt more than they compute.
kernel void fn_qmv_small_b(
    device const uint*   w      [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const float*  x      [[buffer(3)]],   // [nb][in_dim]
    device float*        y      [[buffer(4)]],   // [nb][out_dim]
    constant FnQmvParams& p     [[buffer(5)]],
    constant uint&       nb     [[buffer(6)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint spt   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint g = tgpos * spt + sgid;
    if (g >= nb * p.out_dim) return;
    const uint b = g / p.out_dim;
    const uint row = g % p.out_dim;
    const uint words = p.in_dim / 8;
    device const uint* wr = w + (ulong)row * words;
    device const bfloat* sr = scales + (ulong)row * (p.in_dim / 64);
    device const bfloat* br = biases + (ulong)row * (p.in_dim / 64);
    device const float* xb = x + (ulong)b * p.in_dim;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += 32) {
        const uint word = wr[wi];
        const uint grp = wi >> 3;
        float qd = 0.0f;
        float xs = 0.0f;
        for (uint i = 0; i < 8; i++) {
            const float xv = xb[wi * 8 + i];
            qd = fma((float)((word >> (4 * i)) & 0xFu), xv, qd);
            xs += xv;
        }
        acc = fma((float)sr[grp], qd, fma((float)br[grp], xs, acc));
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(ulong)b * p.out_dim + row] = acc;
}

// x[i] = silu(x[i] / div)
kernel void fn_silu_rows(
    device float*   x   [[buffer(0)]],
    constant uint&  n   [[buffer(1)]],
    constant float& div [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float v = x[i] / div;
    x[i] = v / (1.0f + exp(-v));
}

// out[r] = x[idx[r]] for n_rows rows of `width`
kernel void fn_gather_rows(
    device const float* x     [[buffer(0)]],
    device const uint*  idx   [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant uint&      n_rows [[buffer(3)]],
    constant uint&      width [[buffer(4)]],
    uint gi [[thread_position_in_grid]])
{
    if (gi >= n_rows * width) return;
    const uint r = gi / width;
    const uint d = gi % width;
    out[gi] = x[(ulong)idx[r] * width + d];
}

// out[idx[r]] += w[r] * y[r]; rows of one expert are distinct, so no
// two threads touch the same element.
kernel void fn_scatter_add_rows(
    device const float* y     [[buffer(0)]],
    device const uint*  idx   [[buffer(1)]],
    device const float* w     [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant uint&      n_rows [[buffer(4)]],
    constant uint&      width [[buffer(5)]],
    uint gi [[thread_position_in_grid]])
{
    if (gi >= n_rows * width) return;
    const uint r = gi / width;
    const uint d = gi % width;
    out[(ulong)idx[r] * width + d] += w[r] * y[gi];
}

// out[t] += sigmoid(gate . x[t]) * y[t]; one threadgroup per row.
kernel void fn_shared_add_rows(
    device float*        out   [[buffer(0)]],
    device const float*  y     [[buffer(1)]],
    device const bfloat* gate  [[buffer(2)]],
    device const float*  x     [[buffer(3)]],
    constant uint&       width [[buffer(4)]],
    uint t    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    device const float* xr = x + (ulong)t * width;
    float acc = 0.0f;
    for (uint i = tid; i < width; i += tpg) acc += (float)gate[i] * xr[i];
    acc = simd_sum(acc);
    if (lane == 0) red[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint s = 0; s < (tpg + 31) / 32; s++) dot += red[s];
    const float g = 1.0f / (1.0f + exp(-dot));
    device float* o = out + (ulong)t * width;
    device const float* yr = y + (ulong)t * width;
    for (uint i = tid; i < width; i += tpg) o[i] += g * yr[i];
}
