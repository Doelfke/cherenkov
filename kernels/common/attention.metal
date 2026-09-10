// QK normalization/RoPE, prefill staging, q8 KV storage, and decode attention.

struct QkRopeParams {
    uint n_heads;
    uint head_dim;   // 256
    uint stride;     // elements between consecutive heads' query starts
    uint rot;        // 64
    uint pos;
    float theta;
    float eps;
};

struct AttnGemmParams {
    uint kv_row;          // n_kv * head_dim
    uint blocks_per_row;  // kv_row / 32
    uint head;            // kv head index
    uint kl;              // real cache length (base + nb)
    uint kl_pad;          // kl rounded up to 32
    uint scale_off;       // byte offset of scales plane in kc/vc
    uint base;            // context length before this chunk
    uint nb;              // tokens in this chunk
    uint n_rep;           // q heads per kv head
    float scale;          // 1/sqrt(head_dim), folded into staged q
};

// Dequantize one kv head's K rows ([t][256] half) and V columns
// (transposed: [d][kl_pad] half) from the q8 cache. Rows/cols beyond kl
// are zeroed so downstream GEMMs never see uninitialized memory.
kernel void attn_kv_stage(
    device const char* kc  [[buffer(0)]],
    device const char* vc  [[buffer(1)]],
    device half*       kh  [[buffer(2)]],  // [kl_pad][256]
    device half*       vt  [[buffer(3)]],  // [256][kl_pad]
    constant AttnGemmParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint t = gid / 256;
    const uint d = gid % 256;
    if (t >= p.kl_pad) return;
    if (t >= p.kl) {
        kh[(ulong)t * 256 + d] = 0.0h;
        vt[(ulong)d * p.kl_pad + t] = 0.0h;
        return;
    }
    device const half* ks = (device const half*)(kc + p.scale_off);
    device const half* vs = (device const half*)(vc + p.scale_off);
    const uint e = p.head * 256 + d;
    const uint blk = e / 32;
    float dk = (float)ks[(ulong)t * p.blocks_per_row + blk];
    float dv = (float)vs[(ulong)t * p.blocks_per_row + blk];
    kh[(ulong)t * 256 + d] =
        (half)((float)kc[(ulong)t * p.kv_row + e] * dk);
    vt[(ulong)d * p.kl_pad + t] =
        (half)((float)vc[(ulong)t * p.kv_row + e] * dv);
}

// Stage q rows for ALL heads as half with the softmax scale folded in:
// qh[(head*nb + t)][256] from qg [t][n_heads][2*256] (q half of each
// head's [query|gate] pair).
kernel void attn_q_stage(
    device const float* qg [[buffer(0)]],
    device half*        qh [[buffer(1)]],
    constant AttnGemmParams& p [[buffer(2)]],
    constant uint&      n_heads [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint d = gid % 256;
    const uint t = (gid / 256) % p.nb;
    const uint h = gid / (256 * p.nb);
    if (h >= n_heads) return;
    float q = qg[((ulong)t * n_heads + h) * 512 + d];
    qh[((ulong)h * p.nb + t) * 256 + d] = (half)(q * p.scale);
}

// Scatter one kv head's O rows into attn_out with the sigmoid output
// gate applied: attn_out[t][(head*n_rep+qh)*256 + d].
kernel void attn_o_scatter(
    device const float* O  [[buffer(0)]],   // [(qh*nb + t)][256]
    device const float* qg [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant AttnGemmParams& p [[buffer(3)]],
    constant uint&      n_heads [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint d = gid % 256;
    const uint t = (gid / 256) % p.nb;
    const uint qh = gid / (256 * p.nb);
    if (qh >= p.n_rep) return;
    const uint gh = p.head * p.n_rep + qh;
    float gate = qg[((ulong)t * n_heads + gh) * 512 + 256 + d];
    out[((ulong)t * n_heads + gh) * 256 + d] =
        O[((ulong)qh * p.nb + t) * 256 + d] / (1.0f + exp(-gate));
}

// Batched per-head QK-norm + RoPE. One simdgroup per (token, head);
// x layout [b][n_heads*stride]; position = base_pos + b.
kernel void qk_norm_rope_b(
    device float*        x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    constant QkRopeParams& p [[buffer(2)]],
    constant uint&       nb [[buffer(3)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint spt  [[simdgroups_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint gsg = gid * spt + sg;
    if (gsg >= nb * p.n_heads) return;
    uint b = gsg / p.n_heads;
    uint head = gsg % p.n_heads;
    device float* h = x + (ulong)b * p.n_heads * p.stride + (ulong)head * p.stride;
    const uint per_lane = p.head_dim / 32;
    float ss = 0.0f;
    for (uint j = 0; j < per_lane; j++) {
        float v = h[lane * per_lane + j];
        ss += v * v;
    }
    float total = simd_sum(ss);
    float inv = rsqrt(total / (float)p.head_dim + p.eps);
    for (uint j = 0; j < per_lane; j++) {
        uint idx = lane * per_lane + j;
        h[idx] = h[idx] * inv * (float)w[idx];
    }
    simdgroup_barrier(mem_flags::mem_device);
    uint half_rot = p.rot / 2;
    if (lane < half_rot) {
        float inv_freq = pow(p.theta, -2.0f * (float)lane / (float)p.rot);
        float angle = (float)(p.pos + b) * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        float a = h[lane];
        float bb = h[lane + half_rot];
        h[lane] = a * c - bb * s;
        h[lane + half_rot] = bb * c + a * s;
    }
}

// ---------------- Split-T attention (flash-decode style) ----------------

constant constexpr uint ATTN_TB = 512;  // positions per partial block

struct AttnPartParams {
    uint n_heads;
    uint n_kv;
    uint head_dim;   // 256
    uint t_len;
    uint q_stride;   // 512 (per-head [query|gate])
    uint q_off;      // element offset of this token's q block
    uint max_blk;    // partials stride per head
    float scale;
};

// One threadgroup per head: merge partials (log-sum-exp), apply the
// sigmoid gate, write out[head][head_dim].
kernel void attn_combine(
    device const float* q    [[buffer(0)]],
    device const float* part [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant AttnPartParams& p [[buffer(3)]],
    constant uint&      nblk [[buffer(4)]],
    constant uint&      out_off [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    const uint hd = p.head_dim;
    device const float* ph = part + (ulong)head * p.max_blk * (2 + hd);
    float m = -INFINITY;
    for (uint b = 0; b < nblk; b++) m = max(m, ph[b * (2 + hd)]);
    float s = 0.0f;
    for (uint b = 0; b < nblk; b++) {
        s += exp2(ph[b * (2 + hd)] - m) * ph[b * (2 + hd) + 1];
    }
    float inv_s = 1.0f / s;
    device const float* qh = q + p.q_off + (ulong)head * p.q_stride;
    for (uint d = tid; d < hd; d += tpg) {
        float acc = 0.0f;
        for (uint b = 0; b < nblk; b++) {
            acc += exp2(ph[b * (2 + hd)] - m) * ph[b * (2 + hd) + 2 + d];
        }
        float gate = qh[hd + d];
        out[out_off + (ulong)head * hd + d] =
            acc * inv_s * (1.0f / (1.0f + exp(-gate)));
    }
}

// q8 KV cache: separate quantized-value and scale planes.
// Layout per cache side: int8 quants plane [max_t][1024] then f16 scales
// plane [max_t][32] (one symmetric scale per 32-element block, d = amax/127).
// scale_off = byte offset of the scales plane within the buffer.

struct KvQParams {
    uint row;        // kv_row elements (1024)
    uint t0;         // first position
    uint nb;         // tokens
    uint scale_off;  // byte offset of the scales plane
};

// Quantize-on-append: one simdgroup per 32-element block; lane = element.
// Handles K and V in one dispatch (side = z index of the simdgroup).
kernel void kv_append_q8(
    device const float* k  [[buffer(0)]],  // [nb][row] f32 (post norm+rope)
    device const float* v  [[buffer(1)]],
    device char*        kc [[buffer(2)]],
    device char*        vc [[buffer(3)]],
    constant KvQParams& p  [[buffer(4)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint spt   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint blocks_per_row = p.row / 32;
    const uint gsg = tgpos * spt + sgid;
    const uint total = p.nb * blocks_per_row * 2;  // both sides
    if (gsg >= total) return;
    const uint side = gsg / (p.nb * blocks_per_row);
    const uint rem = gsg % (p.nb * blocks_per_row);
    const uint b = rem / blocks_per_row;
    const uint blk = rem % blocks_per_row;

    device const float* src = (side == 0 ? k : v) + (ulong)b * p.row + blk * 32;
    device char* dstq = (side == 0 ? kc : vc);
    device half* dsts = (device half*)((side == 0 ? kc : vc) + p.scale_off);

    float val = src[lane];
    float amax = simd_max(fabs(val));
    float d = amax / 127.0f;
    float q = d > 0.0f ? rint(val / d) : 0.0f;
    dstq[(ulong)(p.t0 + b) * p.row + blk * 32 + lane] =
        (char)clamp(q, -127.0f, 127.0f);
    if (lane == 0) {
        dsts[(ulong)(p.t0 + b) * blocks_per_row + blk] = (half)d;
    }
}

// GQA lock-step flash-decode partials over the q8 cache: same structure as
// attn_part2, dequant fused into the dot (char4 -> float4 times the block
// scale; lane l covers dims l*4 (block l/8) and (l+32)*4 (block 4+l/8)).
kernel void attn_part2_q8(
    device const float* q     [[buffer(0)]],
    device const char*  kc    [[buffer(1)]],
    device const char*  vc    [[buffer(2)]],
    device float*       part  [[buffer(3)]],
    constant AttnPartParams& p [[buffer(4)]],
    constant uint&      n_wg  [[buffer(5)]],
    constant uint&      scale_off [[buffer(6)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint hd = p.head_dim;
    const uint kv_row = p.n_kv * hd;
    const uint blocks_per_row = kv_row / 32;
    const uint n_rep = p.n_heads / p.n_kv;
    const uint hk = tgpos % p.n_kv;
    const uint iwg = tgpos / p.n_kv;
    const uint head = hk * n_rep + sgid;
    if (sgid >= n_rep) return;

    device const half* ks = (device const half*)(kc + scale_off);
    device const half* vs = (device const half*)(vc + scale_off);
    // This lane's element offsets within the row, and their block indices.
    const uint e0 = hk * hd + lane * 4;
    const uint e1 = hk * hd + (lane + 32) * 4;
    const uint b0 = e0 / 32;
    const uint b1 = e1 / 32;

    device const float* qh = q + p.q_off + (ulong)head * p.q_stride;
    const float4 qa = *(device const float4*)(qh + lane * 4);
    const float4 qb = *(device const float4*)(qh + (lane + 32) * 4);

    float m = -INFINITY;
    float s = 0.0f;
    float4 oa = 0.0f;
    float4 ob = 0.0f;

    const uint n_chunks = (p.t_len + 31) / 32;
    for (uint c = iwg; c < n_chunks; c += n_wg) {
        const uint c0 = c * 32;
        const uint cn = min(p.t_len - c0, 32u);
        for (uint i = 0; i < cn; i++) {
            const ulong t = c0 + i;
            float ds0 = (float)ks[t * blocks_per_row + b0];
            float ds1 = (float)ks[t * blocks_per_row + b1];
            float4 ka = float4(*(device const char4*)(kc + t * kv_row + e0)) * ds0;
            float4 kb = float4(*(device const char4*)(kc + t * kv_row + e1)) * ds1;
            float partial = dot(qa, ka) + dot(qb, kb);
            float sc = simd_sum(partial) * p.scale;
            float mnew = max(m, sc);
            float factor = exp2(m - mnew);
            float e = exp2(sc - mnew);
            float dv0 = (float)vs[t * blocks_per_row + b0];
            float dv1 = (float)vs[t * blocks_per_row + b1];
            float4 va = float4(*(device const char4*)(vc + t * kv_row + e0)) * dv0;
            float4 vb = float4(*(device const char4*)(vc + t * kv_row + e1)) * dv1;
            oa = oa * factor + e * va;
            ob = ob * factor + e * vb;
            s = s * factor + e;
            m = mnew;
        }
    }

    device float* pb = part + ((ulong)head * p.max_blk + iwg) * (2 + hd);
    if (lane == 0) {
        pb[0] = m;
        pb[1] = s;
    }
    *(device float4*)(pb + 2 + lane * 4) = oa;
    *(device float4*)(pb + 2 + (lane + 32) * 4) = ob;
}
