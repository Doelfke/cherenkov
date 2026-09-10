// Token embeddings and greedy argmax reduction.

struct ArgPair {
    float v;
    uint i;
};

// Stage 1: per-threadgroup argmax over a grid-strided slice of logits.
// Lowest index wins ties.
kernel void argmax_partial(
    device const float* logits   [[buffer(0)]],
    device ArgPair*     partials [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    uint tid   [[thread_position_in_grid]],
    uint grid  [[threads_per_grid]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint ltid  [[thread_position_in_threadgroup]],
    uint tpg   [[threads_per_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup ArgPair shm[32];
    float best = -INFINITY;
    uint besti = 0;
    for (uint i = tid; i < n; i += grid) {
        float v = logits[i];
        if (v > best) { best = v; besti = i; }
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(besti, off);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    if (lane == 0) { shm[sgid].v = best; shm[sgid].i = besti; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ltid == 0) {
        for (uint s = 1; s < (tpg + 31) / 32; s++) {
            if (shm[s].v > best || (shm[s].v == best && shm[s].i < besti)) {
                best = shm[s].v;
                besti = shm[s].i;
            }
        }
        partials[tgid].v = best;
        partials[tgid].i = besti;
    }
}

// Stage 2: reduce partials, write the winner into ids[step + 1].
kernel void argmax_final(
    device const ArgPair* partials [[buffer(0)]],
    device uint*          ids      [[buffer(1)]],
    constant uint&        np       [[buffer(2)]],
    constant uint&        step     [[buffer(3)]],
    uint ltid [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup ArgPair shm[32];
    float best = -INFINITY;
    uint besti = 0;
    for (uint i = ltid; i < np; i += tpg) {
        if (partials[i].v > best || (partials[i].v == best && partials[i].i < besti)) {
            best = partials[i].v;
            besti = partials[i].i;
        }
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(besti, off);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    if (lane == 0) { shm[sgid].v = best; shm[sgid].i = besti; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ltid == 0) {
        for (uint s = 1; s < (tpg + 31) / 32; s++) {
            if (shm[s].v > best || (shm[s].v == best && shm[s].i < besti)) {
                best = shm[s].v;
                besti = shm[s].i;
            }
        }
        ids[step + 1] = besti;
    }
}

// ---------------- Batched prefill path (NB tokens per pass) ----------------

// Embed nb rows: ids[i0..i0+nb] -> hidden[b][hsize].
kernel void embed_rows(
    device const uint*   ids    [[buffer(0)]],
    device const uint*   w      [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float*        hidden [[buffer(4)]],
    constant uint&       hsize  [[buffer(5)]],
    constant uint&       i0     [[buffer(6)]],
    constant uint&       nb     [[buffer(7)]],
    uint gi [[thread_position_in_grid]])
{
    if (gi >= nb * hsize) return;
    uint b = gi / hsize;
    uint i = gi % hsize;
    ulong row = ids[i0 + b];
    uint word = w[row * (hsize / 8) + i / 8];
    float q = float((word >> (4 * (i % 8))) & 0xFu);
    float s = scales[row * (hsize / 64) + i / 64];
    float bb = biases[row * (hsize / 64) + i / 64];
    hidden[gi] = fma(s, q, bb);
}
