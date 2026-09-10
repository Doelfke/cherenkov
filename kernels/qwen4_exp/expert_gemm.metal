// Low-bit routed-expert GEMMs: 128 output channels x 8/16/32 token rows.
// Uses the decode stores directly; shared/dense projections remain Q4.
// K is a multiple of 64, output channels a multiple of 8. The caller pads
// token buffers to the tile width, as for the existing Q4 GEMMs. Matrix
// staging/reduction follows those GEMMs; decode's half-dot math is separate.

template <uint BITS, uint TOKENS>
[[kernel]] void fn_expert_qmm(
    device const uint*   w      [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const float*  x      [[buffer(3)]],  // [NB][in_dim]
    device float*        y      [[buffer(4)]],  // [NB][out_dim]
    constant FnQmvParams& p     [[buffer(5)]],
    constant uint&       ntt    [[buffer(6)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint tiitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sa[128 * 32];
    threadgroup half sb[TOKENS * 32];

    const uint r1 = (tgpos % ntt) * TOKENS;
    const uint r0 = (tgpos / ntt) * 128;
    const uint words_per_row = p.in_dim / 32 * BITS;
    constexpr uint TOKEN_FRAGMENTS = TOKENS / 8;
    const uint groups_per_row = p.in_dim / 64;
    // A-staging: two (row, 16-K-chunk) units per thread. Unit u covers
    // row = u / 2 and K-chunk il = u % 2 of the current 32-K slice.
    const uint u0 = tiitg * 2;
    // B-staging assignment: token + which 8-element K-chunk.
    const uint btok = tiitg / 4;
    const uint bky = 8 * (tiitg % 4);
    device const float* yb = x + (ulong)(r1 + btok) * p.in_dim + bky;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[TOKEN_FRAGMENTS];
    simdgroup_float8x8 mc[4 * TOKEN_FRAGMENTS];
    for (short i = 0; i < 4 * TOKEN_FRAGMENTS; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < p.in_dim; loop_k += 32) {
        // Decode the same 32-code records used by the decode kernels.
        // Each staging unit owns 16 consecutive K values, unpacked four
        // even/odd codes at a time. Reconstruct Q4-code midpoints before
        // applying the original bf16 scale and bias, then stage as half.
        half4 de[2][2];
        half4 dodd[2][2];
        float4 xv[2];
        for (uint uu = 0; uu < 2; uu++) {
            const uint unit = u0 + uu;
            const uint ar = unit / 2;
            const uint il0 = unit % 2;
            const uint arow = min(r0 + ar, p.out_dim - 1);
            device const uint* chunk = w + (ulong)arow * words_per_row
                                          + (loop_k / 32) * BITS;
            uint upper = chunk[il0];
            uint lowest = 0;
            if (BITS == 3) lowest = chunk[2] >> (4 * il0);
            float4 s = (float)scales[(ulong)arow * groups_per_row + loop_k / 64];
            float4 b = (float)biases[(ulong)arow * groups_per_row + loop_k / 64];
            for (uint h = 0; h < 2; h++) {
                float4 even = float4(as_type<uchar4>((upper >> (2 * h)) & 0x03030303u));
                float4 odd = float4(as_type<uchar4>((upper >> (2 * (h + 2))) & 0x03030303u));
                if (BITS == 3) {
                    even = 2.0f * even + float4(as_type<uchar4>((lowest >> h) & 0x01010101u));
                    odd = 2.0f * odd + float4(as_type<uchar4>((lowest >> (h + 2)) & 0x01010101u));
                    even = 2.0f * even + 0.5f;
                    odd = 2.0f * odd + 0.5f;
                } else {
                    even = 4.0f * even + 1.5f;
                    odd = 4.0f * odd + 1.5f;
                }
                de[uu][h] = half4(fma(s, even, b));
                dodd[uu][h] = half4(fma(s, odd, b));
            }
        }
        if (btok < TOKENS) {
            xv[0] = *(device const float4*)(yb);
            xv[1] = *(device const float4*)(yb + 4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Swizzled A store: 8x8 blocks, transposed within block (sa laid
        // out as 16 row-blocks x 4 K-blocks of 64 halves each). Even
        // codes land at K = 2i, odd at K = 2i+1 within each 8-half.
        for (uint uu = 0; uu < 2; uu++) {
            const uint unit = u0 + uu;
            const uint ar = unit / 2;
            const uint il0 = unit % 2;
            const uint sy = ar / 8;
            const uint lx = ar % 8;
            for (uint h = 0; h < 2; h++) {
                threadgroup half* base = sa + 64 * (16 * (2 * il0 + h) + sy) + lx;
                for (uint i = 0; i < 4; i++) {
                    base[8 * (2 * i)] = de[uu][h][i];
                    base[8 * (2 * i + 1)] = dodd[uu][h][i];
                }
            }
        }
        // B store: token-major 8x8 blocks.
        if (btok < TOKENS) {
            const uint ib = TOKEN_FRAGMENTS * (tiitg % 4) + btok / 8;
            const uint ly = btok % 8;
            for (uint i = 0; i < 4; i++) {
                *(sb + 64 * ib + 8 * ly + i) = (half)xv[0][i];
                *(sb + 64 * ib + 8 * ly + 4 + i) = (half)xv[1][i];
            }
        }
        yb += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each simdgroup owns 32 output rows and all token fragments.
        threadgroup const half* lsma = sa + 4 * 64 * sgitg;
        threadgroup const half* lsmb = sb;
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < TOKEN_FRAGMENTS; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4 * TOKEN_FRAGMENTS; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 16 * 64;
            lsmb += TOKEN_FRAGMENTS * 64;
        }
    }

    const uint crow = r0 + 32 * sgitg;
    for (short i = 0; i < 4 * TOKEN_FRAGMENTS; i++) {
        if (crow + 8 * (i % 4) + 8 <= p.out_dim) {
            device float* c = y + crow + (ulong)(r1 + 8 * (i / 4)) * p.out_dim;
            simdgroup_store(mc[i], c + 8 * (i % 4), p.out_dim, 0, false);
        }
    }
}

#define FN_EXPERT_QMM_ARGS \
    device const uint*, device const bfloat*, device const bfloat*, \
    device const float*, device float*, constant FnQmvParams&, constant uint&, \
    uint, uint, uint
#define FN_EXPERT_QMM(B, T) \
    template [[host_name("fn_expert_qmm_q" #B "_n" #T)]] [[kernel]] \
    void fn_expert_qmm<B, T>(FN_EXPERT_QMM_ARGS);
FN_EXPERT_QMM(2, 8)
FN_EXPERT_QMM(2, 16)
FN_EXPERT_QMM(2, 32)
FN_EXPERT_QMM(3, 8)
FN_EXPERT_QMM(3, 16)
FN_EXPERT_QMM(3, 32)
#undef FN_EXPERT_QMM
#undef FN_EXPERT_QMM_ARGS
