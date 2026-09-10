// Half-input, float-output GEMM used by prefill attention.

// ---------------- GEMM-attention (prefill) ----------------
// Attention as two dense GEMMs: S = Q K^T, masked row-softmax, O = P V.
// Causality and QSA visibility are applied by fn_attn_softmax_sel, so
// both GEMMs can operate on rectangular tiles.

struct GemmHParams {
    uint m;    // rows of A/C
    uint n;    // cols of C (rounded up to 32; garbage cols never read)
    uint k;    // contraction length
    uint lda;  // row stride of A (elements)
    uint ldb;  // row stride of B (elements)
    uint ldc;  // row stride of C (elements)
};

// C[m][n] = sum_k A[m][k] * B[n][k]; A,B half, C float. 128x32 tile per
// threadgroup (4 simdgroups of 32x32), K-step 32 - the qmm_af4_w
// geometry with direct half staging instead of dequant.
kernel void gemm_hh(
    device const half*   A [[buffer(0)]],
    device const half*   B [[buffer(1)]],
    device float*        C [[buffer(2)]],
    constant GemmHParams& p [[buffer(3)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint tiitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sa[128 * 32];
    threadgroup half sb[32 * 32];
    const uint ntn = p.n / 32;
    const uint rn = (tgpos % ntn) * 32;
    const uint r0 = (tgpos / ntn) * 128;
    const uint u0 = tiitg * 2;
    const uint btok = tiitg / 4;
    const uint bky = 8 * (tiitg % 4);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < p.k; loop_k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint uu = 0; uu < 2; uu++) {
            const uint unit = u0 + uu;
            const uint ar = unit / 2;
            const uint il0 = unit % 2;
            const uint arow = min(r0 + ar, p.m - 1);
            device const half* src = A + (ulong)arow * p.lda + loop_k + il0 * 16;
            const uint sy = ar / 8;
            const uint lx = ar % 8;
            for (uint i = 0; i < 16; i++) {
                const uint sx = 2 * il0 + i / 8;
                *(sa + 64 * (16 * sx + sy) + 8 * (i % 8) + lx) = src[i];
            }
        }
        {
            const uint ib = 4 * (tiitg % 4) + btok / 8;
            const uint ly = btok % 8;
            device const half* src = B + (ulong)(rn + btok) * p.ldb + loop_k + bky;
            for (uint i = 0; i < 8; i++) {
                *(sb + 64 * ib + 8 * ly + i) = src[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * sgitg;
        threadgroup const half* lsmb = sb;
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 16; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 16 * 64;
            lsmb += 4 * 64;
        }
    }

    // mc[i] covers m-block (i % 4) within this sg's 32 rows and n-block
    // (i / 4); fragment (row, col) = (n_local, m_local), so store
    // transposed to land C[m][n] row-major.
    // Store whenever the fragment STARTS in range: C rows are allocated
    // padded past m, and rows >= m are never read downstream, so partial
    // tail fragments write harmless garbage instead of being dropped.
    const uint crow = r0 + 32 * sgitg;
    for (short i = 0; i < 16; i++) {
        const uint m0 = crow + 8 * (i % 4);
        if (m0 < p.m) {
            simdgroup_store(mc[i], C + (ulong)m0 * p.ldc + rn + 8 * (i / 4),
                            p.ldc, 0, true);
        }
    }
}
