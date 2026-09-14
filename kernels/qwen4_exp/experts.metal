// Expert routing, address tables, Q2/Q3 projections, and resident/fetched MoE execution.

#define FN_MAX_NB 6u
#define FN_SLOT_STRIDE 64u

// y[b][r] = sum_c w[r][c] * x[b][c], bf16 weights. One simdgroup per (b, r).
kernel void fn_bf16_matvec_b(
    device const bfloat* w [[buffer(0)]],
    device const float*  x [[buffer(1)]],
    device float*        y [[buffer(2)]],
    constant uint&       rows [[buffer(3)]],
    constant uint&       cols [[buffer(4)]],
    constant uint&       nb   [[buffer(5)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint spt   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint gsg = tgpos * spt + sgid;
    if (gsg >= nb * rows) return;
    const uint b = gsg / rows;
    const uint r = gsg % rows;
    device const bfloat* wr = w + (ulong)r * cols;
    device const float* xb = x + (ulong)b * cols;
    float acc = 0.0f;
    for (uint c = lane; c < cols; c += 32) acc += (float)wr[c] * xb[c];
    acc = simd_sum(acc);
    if (lane == 0) y[gsg] = acc;
}

// Softmax over n router logits per row (n <= 1024, one threadgroup of n
// threads per row), then top-k by repeated argmax; writes indices and the
// renormalized (or raw) probabilities.
kernel void fn_topk_softmax_b(
    device const float* logits_all [[buffer(0)]],  // [nb][n]
    device uint*        idx_all    [[buffer(1)]],  // [nb][k]
    device float*       wts_all    [[buffer(2)]],  // [nb][k]
    constant uint&      n      [[buffer(3)]],
    constant uint&      k      [[buffer(4)]],
    constant uint&      renorm [[buffer(5)]],
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    threadgroup uint redi[32];
    threadgroup float chosen_sum;
    device const float* logits = logits_all + (ulong)b * n;
    device uint* idx = idx_all + (ulong)b * k;
    device float* wts = wts_all + (ulong)b * k;
    const bool live = tid < n;
    float v = live ? logits[tid] : -INFINITY;
    float m = simd_max(v);
    if (lane == 0) red[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = -INFINITY;
    for (uint s = 0; s < (tpg + 31) / 32; s++) m = max(m, red[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = live ? exp(v - m) : 0.0f;
    float sum = simd_sum(e);
    if (lane == 0) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint s = 0; s < (tpg + 31) / 32; s++) sum += red[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float prob = e / sum;
    float cand = live ? prob : -1.0f;
    if (tid == 0) chosen_sum = 0.0f;
    for (uint round = 0; round < k; round++) {
        float bv = cand;
        uint bi = tid;
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(bv, off);
            uint oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red[sgid] = bv; redi[sgid] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float best = red[0];
            uint besti = redi[0];
            for (uint s = 1; s < (tpg + 31) / 32; s++) {
                if (red[s] > best || (red[s] == best && redi[s] < besti)) {
                    best = red[s];
                    besti = redi[s];
                }
            }
            idx[round] = besti;
            wts[round] = best;
            chosen_sum += best;
            redi[31] = besti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == redi[31]) cand = -1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (renorm != 0 && tid == 0) {
        for (uint j = 0; j < k; j++) wts[j] = wts[j] / chosen_sum;
    }
}

// ---- grouped experts over the union of the rows' routed experts ----
// The union is published by the CPU after the router: tab[layer][u] holds
// the GPU address of unique expert u's record (a residency-set member;
// records already resident first, then the ones being fetched),
// tab[layer][62] the resident count, tab[layer][63] the union size, and
// wmap[layer][b][u] each row's routing weight for it (0 when the row did
// not pick it). The dispatch runs in two parts: part 0 covers the
// resident experts plus the shared expert (stored at index p.n_u), part 1
// the late arrivals, which the combine accumulates. Grids are sized for
// the largest union (p.n_u = nb * k); jobs outside their part exit.
struct MoeBParams {
    uint inter;
    uint hidden;
    uint n_u;       // grid bound on unique routed experts (nb * k)
    uint gate_w;    // byte offsets inside a record
    uint up_w;
    uint down_w;
    uint gate_s;
    uint gate_b;
    uint up_s;
    uint up_b;
    uint down_s;
    uint down_b;
    uint layer;     // row of slot_tab / wmap
    uint shared;    // 1: expert n_u is the shared expert, weights in `dense`
    uint nb;
    uint sh_gate_w;
    uint sh_gate_s;
    uint sh_gate_b;
    uint sh_up_w;
    uint sh_up_s;
    uint sh_up_b;
    uint sh_down_w;
    uint sh_down_s;
    uint sh_down_b;
    uint sh_gate_vec;
    uint part;      // 0: resident experts + shared; 1: late arrivals
    // Byte offsets inside a low-bit record (3-bit: 32 codes per 12 bytes;
    // 2-bit: 16 codes per u32; scales and biases as in the 4-bit record).
    // A table entry with bit 63 set points at a 3-bit record, bit 62 at
    // a 2-bit one.
    uint low_gate_w;
    uint low_up_w;
    uint low_down_w;
    uint low_gate_s;
    uint low_gate_b;
    uint low_up_s;
    uint low_up_b;
    uint low_down_s;
    uint low_down_b;
};

// Which unique-expert jobs belong to this part; `u == p.n_u` is the shared
// expert's job in part 0.
static inline bool fn_moe_live(constant MoeBParams& p, device const ulong* tab, uint u, thread bool& is_shared)
{
    const uint n_res = (uint)tab[p.layer * FN_SLOT_STRIDE + FN_SLOT_STRIDE - 2];
    const uint nu = (uint)tab[p.layer * FN_SLOT_STRIDE + FN_SLOT_STRIDE - 1];
    is_shared = (u == p.n_u) && (p.shared != 0);
    if (p.part == 0) return u < n_res || is_shared;
    return u >= n_res && u < nu;
}

// The record of unique expert u, by GPU address (bits 63 and 62 flag
// 3-bit and 2-bit records).
static inline device const uchar* fn_moe_rec(constant MoeBParams& p, device const ulong* tab, uint u)
{
    return (device const uchar*)(tab[p.layer * FN_SLOT_STRIDE + u] & 0x3FFFFFFFFFFFFFFFul);
}

// 0: 4-bit record, 1: 3-bit, 2: 2-bit.
static inline uint fn_moe_kind(constant MoeBParams& p, device const ulong* tab, uint u)
{
    const ulong e = tab[p.layer * FN_SLOT_STRIDE + u];
    return (e >> 63) ? 1u : ((e >> 62) & 1ul ? 2u : 0u);
}

// 2-bit dot of one weight row against NB half-stream rows. Codes are
// q4 >> 2, reconstructed at the quad midpoint: w = (4q + 1.5) s + b, so a
// chunk contributes s (4 sum(q x) + 1.5 sum(x)) + b sum(x).
//
// The store in src/qwen4_exp/lowbit.rs places a word's 16 codes so one
// masked cast yields four consecutive even codes (casts 0 and 1, paired with xe) or
// four consecutive odd ones (casts 2 and 3, paired with xo), the same
// shape as the 4-bit nibble trick. That is eight masked casts and eight
// fma pairs per 32 codes, exactly the 4-bit kernel's instruction count,
// over half the bytes: measured 26 percent faster than 4-bit at nb = 2
// where the code-at-a-time version was 133 percent slower.
template <uint NB>
static inline void fn_q2_rows_h(
    device const uint*   wr,
    device const bfloat* sr,
    device const bfloat* br,
    device const half*   xe,
    device const half*   xo,
    device const float*  xsum,
    uint halves,
    uint hstride,
    uint gstride,
    uint lane,
    thread float* acc)
{
    device const half4* xe4 = (device const half4*)xe;
    device const half4* xo4 = (device const half4*)xo;
    const uint bs4 = hstride / 4;
    device const uint2* w2r = (device const uint2*)wr;
    for (uint hg = lane; hg < halves; hg += 32) {
        const uint2 w2 = w2r[hg];
        const float s = (float)sr[hg >> 1];
        const float bb = (float)br[hg >> 1];
        half4 qd[NB];
        for (uint r = 0; r < NB; r++) qd[r] = 0.0h;
        for (uint t = 0; t < 2; t++) {
            const uint word = (t == 0) ? w2.x : w2.y;
            const half4 e0 = half4(as_type<uchar4>(word & 0x03030303u));
            const half4 e1 = half4(as_type<uchar4>((word >> 2) & 0x03030303u));
            const half4 o0 = half4(as_type<uchar4>((word >> 4) & 0x03030303u));
            const half4 o1 = half4(as_type<uchar4>((word >> 6) & 0x03030303u));
            for (uint r = 0; r < NB; r++) {
                qd[r] = fma(e0, xe4[r * bs4 + hg * 4 + 2 * t], qd[r]);
                qd[r] = fma(e1, xe4[r * bs4 + hg * 4 + 2 * t + 1], qd[r]);
                qd[r] = fma(o0, xo4[r * bs4 + hg * 4 + 2 * t], qd[r]);
                qd[r] = fma(o1, xo4[r * bs4 + hg * 4 + 2 * t + 1], qd[r]);
            }
        }
        for (uint r = 0; r < NB; r++) {
            const float f = (float)qd[r].x + (float)qd[r].y + (float)qd[r].z + (float)qd[r].w;
            const float xs = xsum[r * gstride + hg];
            // s (4 sum(q x) + 1.5 sum(x)) + b sum(x)
            acc[r] = fma(4.0f * s, f, fma(fma(1.5f, s, bb), xs, acc[r]));
        }
    }
    for (uint r = 0; r < NB; r++) acc[r] = simd_sum(acc[r]);
}

// 3-bit dot of one weight row against NB half-stream rows. A 32-code
// chunk is three u32 words (packed by src/qwen4_exp/lowbit.rs): the first
// two hold each code's upper two bits in the 2-bit kernel's order. The
// third holds the lowest bits, positioned so each masked cast yields
// the four bits belonging to the corresponding upper-bit cast.
// Both planes come out four codes at a time: code = 2 * upper + lowest.
// This costs 16 to 22 percent more than the 4-bit kernel
// instead of the 133 percent the code-at-a-time version cost.
//
// Codes are q4 >> 1 and w = (2q + 0.5) s + b, so a chunk contributes
// s (2 sum(q x) + 0.5 sum(x)) + b sum(x).
template <uint NB>
static inline void fn_q3_rows_h(
    device const uint*   wr,
    device const bfloat* sr,
    device const bfloat* br,
    device const half*   xe,
    device const half*   xo,
    device const float*  xsum,
    uint halves,
    uint hstride,
    uint gstride,
    uint lane,
    thread float* acc)
{
    device const half4* xe4 = (device const half4*)xe;
    device const half4* xo4 = (device const half4*)xo;
    const uint bs4 = hstride / 4;
    for (uint hg = lane; hg < halves; hg += 32) {
        const uint upper0 = wr[3 * hg];
        const uint upper1 = wr[3 * hg + 1];
        const uint lowest_bits = wr[3 * hg + 2];
        const float s = (float)sr[hg >> 1];
        const float bb = (float)br[hg >> 1];
        half4 qd[NB];
        for (uint r = 0; r < NB; r++) qd[r] = 0.0h;
        for (uint t = 0; t < 2; t++) {
            const uint word = (t == 0) ? upper0 : upper1;
            for (uint j = 0; j < 4; j++) {
                // Packing keeps the upper two bits and lowest bit in
                // separate planes: code = 2*upper + lowest.
                const half4 upper = half4(as_type<uchar4>((word >> (2 * j)) & 0x03030303u));
                const half4 lowest = half4(as_type<uchar4>((lowest_bits >> (4 * t + j)) & 0x01010101u));
                const half4 code = fma((half)2.0h, upper, lowest);
                const uint idx = hg * 4 + 2 * t + (j & 1);
                for (uint r = 0; r < NB; r++) {
                    qd[r] = fma(code, (j < 2) ? xe4[r * bs4 + idx] : xo4[r * bs4 + idx], qd[r]);
                }
            }
        }
        for (uint r = 0; r < NB; r++) {
            const float f = (float)qd[r].x + (float)qd[r].y + (float)qd[r].z + (float)qd[r].w;
            const float xs = xsum[r * gstride + hg];
            acc[r] = fma(2.0f * s, f, fma(fma(0.5f, s, bb), xs, acc[r]));
        }
    }
    for (uint r = 0; r < NB; r++) acc[r] = simd_sum(acc[r]);
}

// gate_up[u*NB+b][0..inter] = gate_u(x_b), [inter..2*inter] = up_u(x_b).
// One simdgroup per (expert, output row), all NB rows at once.
#define FN_MOE_GATE_UP_B_ARGS                           \
    device const half*  xe   [[buffer(0)]],             \
    device const half*  xo   [[buffer(1)]],             \
    device const float* xsum [[buffer(2)]],             \
    device float*       out  [[buffer(3)]],             \
    constant MoeBParams& p   [[buffer(4)]],             \
    device const ulong* tab  [[buffer(5)]],             \
    device const uchar* dense [[buffer(6)]],            \
    uint tgpos [[threadgroup_position_in_grid]],        \
    uint sgid  [[simdgroup_index_in_threadgroup]],      \
    uint spt   [[simdgroups_per_threadgroup]],          \
    uint lane  [[thread_index_in_simdgroup]]

template <uint NB>
[[kernel]] void fn_moe_gate_up_b(FN_MOE_GATE_UP_B_ARGS)
{
    const uint g = tgpos * spt + sgid;
    const uint rows2 = 2 * p.inter;
    const uint u = g / rows2;
    bool is_shared;
    if (u > p.n_u || !fn_moe_live(p, tab, u, is_shared)) return;
    const uint r = g % rows2;
    const bool up = r >= p.inter;
    const uint row = up ? r - p.inter : r;
    device const uchar* wb;
    device const uchar* sb;
    device const uchar* bb;
    const uint kind = is_shared ? 0u : fn_moe_kind(p, tab, u);
    if (!is_shared) {
        device const uchar* rec = fn_moe_rec(p, tab, u);
        if (kind != 0) {
            wb = rec + (up ? p.low_up_w : p.low_gate_w);
            sb = rec + (up ? p.low_up_s : p.low_gate_s);
            bb = rec + (up ? p.low_up_b : p.low_gate_b);
        } else {
            wb = rec + (up ? p.up_w : p.gate_w);
            sb = rec + (up ? p.up_s : p.gate_s);
            bb = rec + (up ? p.up_b : p.gate_b);
        }
    } else {
        wb = dense + (up ? p.sh_up_w : p.sh_gate_w);
        sb = dense + (up ? p.sh_up_s : p.sh_gate_s);
        bb = dense + (up ? p.sh_up_b : p.sh_gate_b);
    }
    const uint halves = p.hidden / 32;
    device const bfloat* sr = (device const bfloat*)sb + (ulong)row * (halves / 2);
    device const bfloat* br = (device const bfloat*)bb + (ulong)row * (halves / 2);
    float acc[NB];
    for (uint i = 0; i < NB; i++) acc[i] = 0.0f;
    if (kind == 1) {
        device const uint* wr = (device const uint*)wb + (ulong)row * halves * 3;
        fn_q3_rows_h<NB>(wr, sr, br, xe, xo, xsum, halves, p.hidden / 2, halves, lane, acc);
    } else if (kind == 2) {
        device const uint* wr = (device const uint*)wb + (ulong)row * halves * 2;
        fn_q2_rows_h<NB>(wr, sr, br, xe, xo, xsum, halves, p.hidden / 2, halves, lane, acc);
    } else {
        device const uint4* wr = (device const uint4*)wb + (ulong)row * halves;
        fn_q4_rows_h<NB>(wr, sr, br, (device const half4*)xe, (device const half4*)xo, xsum,
                         halves, p.hidden / 8, halves, lane, acc);
    }
    if (lane == 0) {
        for (uint b = 0; b < NB; b++) out[((ulong)u * NB + b) * rows2 + r] = acc[b];
    }
}
#define FN_INST_MOE_GATE_UP(N) \
    template [[host_name("fn_moe_gate_up_b" #N)]] [[kernel]] void fn_moe_gate_up_b<N>(FN_MOE_GATE_UP_B_ARGS);
FN_INST_MOE_GATE_UP(1)
FN_INST_MOE_GATE_UP(2)
FN_INST_MOE_GATE_UP(3)
FN_INST_MOE_GATE_UP(4)
FN_INST_MOE_GATE_UP(5)
FN_INST_MOE_GATE_UP(6)

// h[u*nb+b][j] = silu(gate) * up, written as half even/odd streams plus
// per-32 group sums for the down projection. Grid = (n_u+shared)*nb*inter/2
// threads (a multiple of 16: inter/2 % 16 == 0).
kernel void fn_moe_act_b(
    device const float* gate_up [[buffer(0)]],
    device half*        xe2     [[buffer(1)]],
    device half*        xo2     [[buffer(2)]],
    device float*       xsum2   [[buffer(3)]],
    constant MoeBParams& p      [[buffer(4)]],
    device const ulong* tab     [[buffer(5)]],
    uint gi   [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint half_i = p.inter / 2;
    const uint ub = gi / half_i;
    const uint i = gi % half_i;
    const uint u = ub / p.nb;
    bool is_shared;
    const bool live = u <= p.n_u && fn_moe_live(p, tab, u, is_shared);
    float e = 0.0f;
    float o = 0.0f;
    if (live) {
        device const float* gu = gate_up + (ulong)ub * 2 * p.inter;
        float g0 = gu[2 * i], g1 = gu[2 * i + 1];
        float u0 = gu[p.inter + 2 * i], u1 = gu[p.inter + 2 * i + 1];
        e = (g0 / (1.0f + exp(-g0))) * u0;
        o = (g1 / (1.0f + exp(-g1))) * u1;
        xe2[gi] = (half)e;
        xo2[gi] = (half)o;
    }
    float s = (float)(half)e + (float)(half)o;
    s += simd_shuffle_xor(s, 1);
    s += simd_shuffle_xor(s, 2);
    s += simd_shuffle_xor(s, 4);
    s += simd_shuffle_xor(s, 8);
    if (live && (lane & 15) == 0) xsum2[ub * (p.inter / 32) + i / 16] = s;
}

// y[u*NB+b][row] = down_u(h[u*NB+b]). One simdgroup per (expert, row).
#define FN_MOE_DOWN_B_ARGS                              \
    device const half*  xe2   [[buffer(0)]],            \
    device const half*  xo2   [[buffer(1)]],            \
    device const float* xsum2 [[buffer(2)]],            \
    device float*       y     [[buffer(3)]],            \
    constant MoeBParams& p    [[buffer(4)]],            \
    device const ulong* tab   [[buffer(5)]],            \
    device const uchar* dense [[buffer(6)]],            \
    uint tgpos [[threadgroup_position_in_grid]],        \
    uint sgid  [[simdgroup_index_in_threadgroup]],      \
    uint spt   [[simdgroups_per_threadgroup]],          \
    uint lane  [[thread_index_in_simdgroup]]

// One simdgroup computes TWO consecutive output rows of one expert
// (p.hidden is even), sharing the x loads: the down projection's rows are
// only 20 half-groups long, so pairing them is 26 percent faster at the
// same numerics.
template <uint NB>
[[kernel]] void fn_moe_down_b(FN_MOE_DOWN_B_ARGS)
{
    const uint half_h = p.hidden / 2;
    const uint g = tgpos * spt + sgid;
    const uint u = g / half_h;
    bool is_shared;
    if (u > p.n_u || !fn_moe_live(p, tab, u, is_shared)) return;
    const uint row = 2 * (g % half_h);
    device const uchar* wb;
    device const uchar* sb;
    device const uchar* bb;
    const uint kind = is_shared ? 0u : fn_moe_kind(p, tab, u);
    if (!is_shared) {
        device const uchar* rec = fn_moe_rec(p, tab, u);
        if (kind != 0) {
            wb = rec + p.low_down_w;
            sb = rec + p.low_down_s;
            bb = rec + p.low_down_b;
        } else {
            wb = rec + p.down_w;
            sb = rec + p.down_s;
            bb = rec + p.down_b;
        }
    } else {
        wb = dense + p.sh_down_w;
        sb = dense + p.sh_down_s;
        bb = dense + p.sh_down_b;
    }
    const uint halves = p.inter / 32;
    const ulong xrow = (ulong)u * NB;
    float acc[NB];
    float acc1[NB];
    for (uint i = 0; i < NB; i++) { acc[i] = 0.0f; acc1[i] = 0.0f; }
    for (uint j = 0; j < 2; j++) {
        const uint rj = row + j;
        device const bfloat* sr = (device const bfloat*)sb + (ulong)rj * (halves / 2);
        device const bfloat* br = (device const bfloat*)bb + (ulong)rj * (halves / 2);
        thread float* a = (j == 0) ? acc : acc1;
        if (kind == 1) {
            device const uint* wr = (device const uint*)wb + (ulong)rj * halves * 3;
            fn_q3_rows_h<NB>(wr, sr, br,
                             xe2 + xrow * (p.inter / 2),
                             xo2 + xrow * (p.inter / 2),
                             xsum2 + xrow * halves,
                             halves, p.inter / 2, halves, lane, a);
        } else if (kind == 2) {
            device const uint* wr = (device const uint*)wb + (ulong)rj * halves * 2;
            fn_q2_rows_h<NB>(wr, sr, br,
                             xe2 + xrow * (p.inter / 2),
                             xo2 + xrow * (p.inter / 2),
                             xsum2 + xrow * halves,
                             halves, p.inter / 2, halves, lane, a);
        } else if (kind == 0 && j == 0) {
            // Both rows at once, sharing the x loads.
            device const uint4* wr0 = (device const uint4*)wb + (ulong)row * halves;
            device const uint4* wr1 = wr0 + halves;
            device const bfloat* sr1 = sr + (halves / 2);
            device const bfloat* br1 = br + (halves / 2);
            fn_q4_2rows_h<NB>(wr0, wr1, sr, br, sr1, br1,
                              (device const half4*)xe2 + xrow * (p.inter / 8),
                              (device const half4*)xo2 + xrow * (p.inter / 8),
                              xsum2 + xrow * halves,
                              halves, p.inter / 8, halves, lane, acc, acc1);
        }
        if (kind == 0) break;
    }
    if (lane == 0) {
        for (uint b = 0; b < NB; b++) {
            y[(xrow + b) * p.hidden + row] = acc[b];
            y[(xrow + b) * p.hidden + row + 1] = acc1[b];
        }
    }
}
#define FN_INST_MOE_DOWN(N) \
    template [[host_name("fn_moe_down_b" #N)]] [[kernel]] void fn_moe_down_b<N>(FN_MOE_DOWN_B_ARGS);
FN_INST_MOE_DOWN(1)
FN_INST_MOE_DOWN(2)
FN_INST_MOE_DOWN(3)
FN_INST_MOE_DOWN(4)
FN_INST_MOE_DOWN(5)
FN_INST_MOE_DOWN(6)

// Part 0: out[b][i] = sum_{u < n_res} wmap[layer][b][u] * y[u*nb+b][i]
//                   + sigmoid(gate_vec . x[b]) * y[n_u*nb+b][i]
// Part 1: out[b][i] += sum_{n_res <= u < nu} wmap[layer][b][u] * y[u*nb+b][i]
// One threadgroup per (row, 256 outputs); the gate dot is recomputed per
// threadgroup.
kernel void fn_moe_combine_b(
    device const float* y    [[buffer(0)]],
    device const float* wmap [[buffer(1)]],   // [layer][FN_MAX_NB][FN_SLOT_STRIDE]
    device float*       out  [[buffer(2)]],
    constant MoeBParams& p   [[buffer(3)]],
    device const uchar* dense [[buffer(4)]],
    device const float* x    [[buffer(5)]],   // [nb][hidden]
    device const ulong* tab  [[buffer(6)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    const uint nchunk = (p.hidden + 255) / 256;
    const uint b = tg / nchunk;
    const uint i = (tg % nchunk) * 256 + tid;
    const uint n_res = (uint)tab[p.layer * FN_SLOT_STRIDE + FN_SLOT_STRIDE - 2];
    const uint nu = (uint)tab[p.layer * FN_SLOT_STRIDE + FN_SLOT_STRIDE - 1];
    if (p.part != 0) {
        if (i >= p.hidden) return;
        device const float* wm = wmap + ((ulong)p.layer * FN_MAX_NB + b) * FN_SLOT_STRIDE;
        float acc = 0.0f;
        for (uint u = n_res; u < nu; u++) {
            const float wv = wm[u];
            if (wv != 0.0f) acc = fma(wv, y[((ulong)u * p.nb + b) * p.hidden + i], acc);
        }
        out[(ulong)b * p.hidden + i] += acc;
        return;
    }
    float gs = 0.0f;
    if (p.shared != 0) {
        device const bfloat* gv = (device const bfloat*)(dense + p.sh_gate_vec);
        device const float* xb = x + (ulong)b * p.hidden;
        float acc = 0.0f;
        for (uint j = tid; j < p.hidden; j += tpg) acc += (float)gv[j] * xb[j];
        acc = simd_sum(acc);
        if (lane == 0) red[sgid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dot = 0.0f;
        for (uint s = 0; s < (tpg + 31) / 32; s++) dot += red[s];
        gs = 1.0f / (1.0f + exp(-dot));
    }
    if (i >= p.hidden) return;
    device const float* wm = wmap + ((ulong)p.layer * FN_MAX_NB + b) * FN_SLOT_STRIDE;
    float acc = 0.0f;
    for (uint u = 0; u < n_res; u++) {
        const float wv = wm[u];
        if (wv != 0.0f) acc = fma(wv, y[((ulong)u * p.nb + b) * p.hidden + i], acc);
    }
    if (p.shared != 0) acc = fma(gs, y[((ulong)p.n_u * p.nb + b) * p.hidden + i], acc);
    out[(ulong)b * p.hidden + i] = acc;
}
