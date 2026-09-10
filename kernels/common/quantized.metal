// Affine 4-bit dense projections: prefill GEMMs, verify matvecs, and input staging.

constant constexpr uint GROUP_SIZE = 64;

struct QmvParams {
    uint out_dim;
    uint in_dim;
};

// Affine-Q4 GEMM: 128 output rows x 32 token rows x 32 K per
// threadgroup (128 threads). Four simdgroups each own a 32x32 output
// tile. Device loads stage in registers before reusing the shared half
// tiles; barriers separate staging from matrix work. Output edges are
// clamp-loaded and guard-stored. qmm_from passes ntt = ceil(nb/32) and
// pads input rows so unused tile lanes cannot read beyond the buffer.
kernel void qmm_af4_w(
    device const uint*   w      [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const float*  x      [[buffer(3)]],  // [NB][in_dim]
    device float*        y      [[buffer(4)]],  // [NB][out_dim]
    constant QmvParams&  p      [[buffer(5)]],
    constant uint&       ntt    [[buffer(6)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint tiitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sa[128 * 32];
    threadgroup half sb[32 * 32];

    const uint r1 = (tgpos % ntt) * 32;
    const uint r0 = (tgpos / ntt) * 128;
    const uint words_per_row = p.in_dim / 8;
    const uint groups_per_row = p.in_dim / GROUP_SIZE;
    // A-staging: two (row, 16-K-chunk) units per thread. Unit u covers
    // row = u / 2 and K-chunk il = u % 2 of the current 32-K slice.
    const uint u0 = tiitg * 2;
    // B-staging assignment: token + which 8-element K-chunk.
    const uint btok = tiitg / 4;
    const uint bky = 8 * (tiitg % 4);
    device const float* yb = x + (ulong)(r1 + btok) * p.in_dim + bky;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < p.in_dim; loop_k += 32) {
        // Dequant this thread's two staging units into registers before the
        // barrier so device loads overlap the previous MMA phase. Bytewise
        // nibble split: uchar4 of (w & 0x0F0F0F0F) yields K positions
        // 0,2,4,6 (even) and the >>4 half yields 1,3,5,7 (odd); the
        // scattered swizzle store remaps them to sequential K.
        half4 de[2][2];
        half4 dodd[2][2];
        float4 xv[2];
        for (uint uu = 0; uu < 2; uu++) {
            const uint unit = u0 + uu;
            const uint ar = unit / 2;
            const uint il0 = unit % 2;
            const uint arow = min(r0 + ar, p.out_dim - 1);
            uint2 w2 = *((device const uint2*)(w + (ulong)arow * words_per_row
                                               + loop_k / 8) + il0);
            float s = (float)scales[(ulong)arow * groups_per_row + loop_k / GROUP_SIZE];
            float b = (float)biases[(ulong)arow * groups_per_row + loop_k / GROUP_SIZE];
            float4 s4 = s;
            float4 b4 = b;
            de[uu][0] = half4(fma(s4, float4(as_type<uchar4>(w2.x & 0x0F0F0F0Fu)), b4));
            dodd[uu][0] =
                half4(fma(s4, float4(as_type<uchar4>((w2.x >> 4) & 0x0F0F0F0Fu)), b4));
            de[uu][1] = half4(fma(s4, float4(as_type<uchar4>(w2.y & 0x0F0F0F0Fu)), b4));
            dodd[uu][1] =
                half4(fma(s4, float4(as_type<uchar4>((w2.y >> 4) & 0x0F0F0F0Fu)), b4));
        }
        xv[0] = *(device const float4*)(yb);
        xv[1] = *(device const float4*)(yb + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Swizzled A store: 8x8 blocks, transposed within block (sa laid
        // out as 16 row-blocks x 4 K-blocks of 64 halves each). Even
        // nibbles land at K = 2i, odd at K = 2i+1 within each 8-half.
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
        {
            const uint ib = 4 * (tiitg % 4) + btok / 8;
            const uint ly = btok % 8;
            for (uint i = 0; i < 4; i++) {
                *(sb + 64 * ib + 8 * ly + i) = (half)xv[0][i];
                *(sb + 64 * ib + 8 * ly + 4 + i) = (half)xv[1][i];
            }
        }
        yb += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each simdgroup: rows sgitg*32..+32, all 32 tokens.
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

    const uint crow = r0 + 32 * sgitg;
    for (short i = 0; i < 16; i++) {
        if (crow + 8 * (i % 4) + 8 <= p.out_dim) {
            device float* c = y + crow + (ulong)(r1 + 8 * (i / 4)) * p.out_dim;
            simdgroup_store(mc[i], c + 8 * (i % 4), p.out_dim, 0, false);
        }
    }
}

// Small-batch qmm (nb <= 8, one 8-token tile): avoids computing the
// padding of a full 32-token tile. Same A-staging/dequant as qmm_af4_w;
// B shrinks to one 8x8 block per K-chunk and each simdgroup keeps four
// accumulator fragments (32 output rows x 8 tokens).
kernel void qmm_af4_n8(
    device const uint*   w      [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const float*  x      [[buffer(3)]],  // [NB][in_dim]
    device float*        y      [[buffer(4)]],  // [NB][out_dim]
    constant QmvParams&  p      [[buffer(5)]],
    constant uint&       ntt    [[buffer(6)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint tiitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sa[128 * 32];
    threadgroup half sb[8 * 32];

    const uint r1 = (tgpos % ntt) * 8;
    const uint r0 = (tgpos / ntt) * 128;
    const uint words_per_row = p.in_dim / 8;
    const uint groups_per_row = p.in_dim / GROUP_SIZE;
    const uint u0 = tiitg * 2;
    // B-staging: threads 0..31 own (token btok, 8-K-chunk bky).
    const uint btok = tiitg / 4;
    const uint bky = 8 * (tiitg % 4);
    device const float* yb = x + (ulong)(r1 + btok) * p.in_dim + bky;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc[4];
    for (short i = 0; i < 4; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < p.in_dim; loop_k += 32) {
        half4 de[2][2];
        half4 dodd[2][2];
        float4 xv[2];
        for (uint uu = 0; uu < 2; uu++) {
            const uint unit = u0 + uu;
            const uint ar = unit / 2;
            const uint il0 = unit % 2;
            const uint arow = min(r0 + ar, p.out_dim - 1);
            uint2 w2 = *((device const uint2*)(w + (ulong)arow * words_per_row
                                               + loop_k / 8) + il0);
            float s = (float)scales[(ulong)arow * groups_per_row + loop_k / GROUP_SIZE];
            float b = (float)biases[(ulong)arow * groups_per_row + loop_k / GROUP_SIZE];
            float4 s4 = s;
            float4 b4 = b;
            de[uu][0] = half4(fma(s4, float4(as_type<uchar4>(w2.x & 0x0F0F0F0Fu)), b4));
            dodd[uu][0] =
                half4(fma(s4, float4(as_type<uchar4>((w2.x >> 4) & 0x0F0F0F0Fu)), b4));
            de[uu][1] = half4(fma(s4, float4(as_type<uchar4>(w2.y & 0x0F0F0F0Fu)), b4));
            dodd[uu][1] =
                half4(fma(s4, float4(as_type<uchar4>((w2.y >> 4) & 0x0F0F0F0Fu)), b4));
        }
        if (btok < 8) {
            xv[0] = *(device const float4*)(yb);
            xv[1] = *(device const float4*)(yb + 4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
        if (btok < 8) {
            const uint ib = tiitg % 4;
            const uint ly = btok;
            for (uint i = 0; i < 4; i++) {
                *(sb + 64 * ib + 8 * ly + i) = (half)xv[0][i];
                *(sb + 64 * ib + 8 * ly + 4 + i) = (half)xv[1][i];
            }
        }
        yb += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * sgitg;
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, sb + 64 * ik, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
            }
            lsma += 16 * 64;
        }
    }

    const uint crow = r0 + 32 * sgitg;
    for (short i = 0; i < 4; i++) {
        if (crow + 8 * i + 8 <= p.out_dim) {
            device float* c = y + crow + (ulong)r1 * p.out_dim;
            simdgroup_store(mc[i], c + 8 * i, p.out_dim, 0, false);
        }
    }
}

// 16-token sibling of qmm_af4_n8 (one 16-token tile, 2 B fragments per
// K-chunk, 8 accumulator fragments per simdgroup): streams the weights
// once for 9–16 rows instead of twice via two 8-token tiles.
kernel void qmm_af4_n16(
    device const uint*   w      [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const float*  x      [[buffer(3)]],  // [NB][in_dim]
    device float*        y      [[buffer(4)]],  // [NB][out_dim]
    constant QmvParams&  p      [[buffer(5)]],
    constant uint&       ntt    [[buffer(6)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint tiitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sa[128 * 32];
    threadgroup half sb[16 * 32];

    const uint r1 = (tgpos % ntt) * 16;
    const uint r0 = (tgpos / ntt) * 128;
    const uint words_per_row = p.in_dim / 8;
    const uint groups_per_row = p.in_dim / GROUP_SIZE;
    const uint u0 = tiitg * 2;
    const uint btok = tiitg / 4;
    const uint bky = 8 * (tiitg % 4);
    device const float* yb = x + (ulong)(r1 + btok) * p.in_dim + bky;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < p.in_dim; loop_k += 32) {
        half4 de[2][2];
        half4 dodd[2][2];
        float4 xv[2];
        for (uint uu = 0; uu < 2; uu++) {
            const uint unit = u0 + uu;
            const uint ar = unit / 2;
            const uint il0 = unit % 2;
            const uint arow = min(r0 + ar, p.out_dim - 1);
            uint2 w2 = *((device const uint2*)(w + (ulong)arow * words_per_row
                                               + loop_k / 8) + il0);
            float s = (float)scales[(ulong)arow * groups_per_row + loop_k / GROUP_SIZE];
            float b = (float)biases[(ulong)arow * groups_per_row + loop_k / GROUP_SIZE];
            float4 s4 = s;
            float4 b4 = b;
            de[uu][0] = half4(fma(s4, float4(as_type<uchar4>(w2.x & 0x0F0F0F0Fu)), b4));
            dodd[uu][0] =
                half4(fma(s4, float4(as_type<uchar4>((w2.x >> 4) & 0x0F0F0F0Fu)), b4));
            de[uu][1] = half4(fma(s4, float4(as_type<uchar4>(w2.y & 0x0F0F0F0Fu)), b4));
            dodd[uu][1] =
                half4(fma(s4, float4(as_type<uchar4>((w2.y >> 4) & 0x0F0F0F0Fu)), b4));
        }
        if (btok < 16) {
            xv[0] = *(device const float4*)(yb);
            xv[1] = *(device const float4*)(yb + 4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
        if (btok < 16) {
            // Blocks: ib = 2*kc + tokgroup(btok/8); [token%8][K elem].
            const uint ib = 2 * (tiitg % 4) + btok / 8;
            const uint ly = btok % 8;
            for (uint i = 0; i < 4; i++) {
                *(sb + 64 * ib + 8 * ly + i) = (half)xv[0][i];
                *(sb + 64 * ib + 8 * ly + 4 + i) = (half)xv[1][i];
            }
        }
        yb += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4 * 64 * sgitg;
        threadgroup const half* lsmb = sb;
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 16 * 64;
            lsmb += 2 * 64;
        }
    }

    const uint crow = r0 + 32 * sgitg;
    for (short i = 0; i < 8; i++) {
        if (crow + 8 * (i % 4) + 8 <= p.out_dim) {
            device float* c = y + crow + (ulong)(r1 + 8 * (i / 4)) * p.out_dim;
            simdgroup_store(mc[i], c + 8 * (i % 4), p.out_dim, 0, false);
        }
    }
}

// Deinterleave x into HALF even/odd streams for the half-math verify qmv,
// and emit per-32-element-group sums (in float) so the qmv bias term reads
// one value per group instead of re-summing x per output row. Group g of
// stream b covers pair-indices [g*16, g*16+16); n2 is always a multiple of
// 32, so each 16-lane simd half maps to exactly one group.
kernel void deinterleave_bh(
    device const float* x    [[buffer(0)]],
    device half*        xe   [[buffer(1)]],
    device half*        xo   [[buffer(2)]],
    constant uint&      n2   [[buffer(3)]],
    constant uint&      nb   [[buffer(4)]],
    device float*       xsum [[buffer(5)]],
    uint gi   [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (gi >= nb * n2) return;
    uint b = gi / n2;
    uint i = gi % n2;
    device const float* xb = x + (ulong)b * n2 * 2;
    float e = xb[2 * i];
    float o = xb[2 * i + 1];
    xe[gi] = (half)e;
    xo[gi] = (half)o;
    float s = e + o;
    s += simd_shuffle_xor(s, 1);
    s += simd_shuffle_xor(s, 2);
    s += simd_shuffle_xor(s, 4);
    s += simd_shuffle_xor(s, 8);
    if ((lane & 15) == 0) {
        xsum[b * (n2 / 16) + i / 16] = s;
    }
}

// One-to-three-row verify qmv with half dot math: per
// half-group, half4 accumulation promoted into f32 accumulators.
kernel void qmv_multi_h(
    device const uint*   w      [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half*   xe     [[buffer(3)]],
    device const half*   xo     [[buffer(4)]],
    device float*        y      [[buffer(5)]],  // [b][out_dim]
    constant QmvParams&  p      [[buffer(6)]],
    constant uint&       nb     [[buffer(7)]],
    device const float*  xsum   [[buffer(8)]],  // [b][in_dim/32] group sums
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint spt   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgpos * spt + sgid;
    if (row >= p.out_dim) return;
    const uint words_per_row = p.in_dim / 8;
    const uint halves_per_row = p.in_dim / 32;
    const uint bstride4 = p.in_dim / 8;  // half4s per token stream
    device const uint4* wr = (device const uint4*)(w + (ulong)row * words_per_row);
    device const bfloat* sr = scales + (ulong)row * (halves_per_row / 2);
    device const bfloat* br = biases + (ulong)row * (halves_per_row / 2);
    device const half4* xe0 = (device const half4*)(xe);
    device const half4* xo0 = (device const half4*)(xo);
    device const half4* xe1 = xe0 + bstride4;
    device const half4* xo1 = xo0 + bstride4;
    device const half4* xe2 = xe1 + bstride4;
    device const half4* xo2 = xo1 + bstride4;
    device const float* xg0 = xsum;
    device const float* xg1 = xsum + halves_per_row;
    device const float* xg2 = xsum + 2 * halves_per_row;
    const bool two = nb > 1;
    const bool three = nb > 2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (uint hg = lane; hg < halves_per_row; hg += 32) {
        uint4 w4 = wr[hg];
        float s = (float)sr[hg >> 1];
        float b = (float)br[hg >> 1];
        half4 qd0 = 0.0h;
        half4 qd1 = 0.0h;
        half4 qd2 = 0.0h;
        uint word;
        half4 lo4;
        half4 hi4;
        half4 xa;
        half4 xb;
        #define QMH_WORD(W, J)                                             \
            word = (W);                                                    \
            lo4 = half4(as_type<uchar4>(word & 0x0F0F0F0Fu));              \
            hi4 = half4(as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu));      \
            xa = xe0[hg * 4 + (J)];                                        \
            xb = xo0[hg * 4 + (J)];                                        \
            qd0 = fma(lo4, xa, qd0);                                       \
            qd0 = fma(hi4, xb, qd0);                                       \
            if (two) {                                                     \
                xa = xe1[hg * 4 + (J)];                                    \
                xb = xo1[hg * 4 + (J)];                                    \
                qd1 = fma(lo4, xa, qd1);                                   \
                qd1 = fma(hi4, xb, qd1);                                   \
            }                                                              \
            if (three) {                                                   \
                xa = xe2[hg * 4 + (J)];                                    \
                xb = xo2[hg * 4 + (J)];                                    \
                qd2 = fma(lo4, xa, qd2);                                   \
                qd2 = fma(hi4, xb, qd2);                                   \
            }
        QMH_WORD(w4.x, 0) QMH_WORD(w4.y, 1)
        QMH_WORD(w4.z, 2) QMH_WORD(w4.w, 3)
        #undef QMH_WORD
        float qdf0 = (float)qd0.x + (float)qd0.y + (float)qd0.z + (float)qd0.w;
        acc0 = fma(s, qdf0, fma(b, xg0[hg], acc0));
        if (two) {
            float qdf1 = (float)qd1.x + (float)qd1.y + (float)qd1.z + (float)qd1.w;
            acc1 = fma(s, qdf1, fma(b, xg1[hg], acc1));
        }
        if (three) {
            float qdf2 = (float)qd2.x + (float)qd2.y + (float)qd2.z + (float)qd2.w;
            acc2 = fma(s, qdf2, fma(b, xg2[hg], acc2));
        }
    }
    acc0 = simd_sum(acc0);
    if (lane == 0) y[row] = acc0;
    if (two) {
        acc1 = simd_sum(acc1);
        if (lane == 0) y[(ulong)p.out_dim + row] = acc1;
    }
    if (three) {
        acc2 = simd_sum(acc2);
        if (lane == 0) y[2 * (ulong)p.out_dim + row] = acc2;
    }
}

// N-stream verify qmv (nb <= 8): one weight stream shared across up to
// 8 token rows. Per-row reduction order is identical to qmv_multi_h /
// qmv (same hg walk, same fma order), so verify numerics match the
// historically gated path exactly. Runtime row guards; ~40 GPRs.
kernel void qmv_multi_hn(
    device const uint*   w      [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half*   xe     [[buffer(3)]],
    device const half*   xo     [[buffer(4)]],
    device float*        y      [[buffer(5)]],  // [b][out_dim]
    constant QmvParams&  p      [[buffer(6)]],
    constant uint&       nb     [[buffer(7)]],
    device const float*  xsum   [[buffer(8)]],  // [b][in_dim/32] group sums
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint spt   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgpos * spt + sgid;
    if (row >= p.out_dim) return;
    const uint words_per_row = p.in_dim / 8;
    const uint halves_per_row = p.in_dim / 32;
    const uint bstride4 = p.in_dim / 8;  // half4s per token stream
    device const uint4* wr = (device const uint4*)(w + (ulong)row * words_per_row);
    device const bfloat* sr = scales + (ulong)row * (halves_per_row / 2);
    device const bfloat* br = biases + (ulong)row * (halves_per_row / 2);
    device const half4* xe0 = (device const half4*)(xe);
    device const half4* xo0 = (device const half4*)(xo);
    const bool r1 = nb > 1;
    const bool r2 = nb > 2;
    const bool r3 = nb > 3;
    const bool r4 = nb > 4;
    const bool r5 = nb > 5;
    const bool r6 = nb > 6;
    const bool r7 = nb > 7;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (uint hg = lane; hg < halves_per_row; hg += 32) {
        uint4 w4 = wr[hg];
        float s = (float)sr[hg >> 1];
        float b = (float)br[hg >> 1];
        half4 qd0 = 0.0h;
        half4 qd1 = 0.0h;
        half4 qd2 = 0.0h;
        half4 qd3 = 0.0h;
        half4 qd4 = 0.0h;
        half4 qd5 = 0.0h;
        half4 qd6 = 0.0h;
        half4 qd7 = 0.0h;
        uint word;
        half4 lo4;
        half4 hi4;
        half4 xa;
        half4 xb;
        #define QMN_WORD(W, J)                                             \
            word = (W);                                                    \
            lo4 = half4(as_type<uchar4>(word & 0x0F0F0F0Fu));              \
            hi4 = half4(as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu));      \
            xa = xe0[hg * 4 + (J)];                                        \
            xb = xo0[hg * 4 + (J)];                                        \
            qd0 = fma(lo4, xa, qd0);                                       \
            qd0 = fma(hi4, xb, qd0); \
            if (r1) { \
                xa = xe0[1u * bstride4 + hg * 4 + (J)]; \
                xb = xo0[1u * bstride4 + hg * 4 + (J)]; \
                qd1 = fma(lo4, xa, qd1); \
                qd1 = fma(hi4, xb, qd1); \
            } \
            if (r2) { \
                xa = xe0[2u * bstride4 + hg * 4 + (J)]; \
                xb = xo0[2u * bstride4 + hg * 4 + (J)]; \
                qd2 = fma(lo4, xa, qd2); \
                qd2 = fma(hi4, xb, qd2); \
            } \
            if (r3) { \
                xa = xe0[3u * bstride4 + hg * 4 + (J)]; \
                xb = xo0[3u * bstride4 + hg * 4 + (J)]; \
                qd3 = fma(lo4, xa, qd3); \
                qd3 = fma(hi4, xb, qd3); \
            } \
            if (r4) { \
                xa = xe0[4u * bstride4 + hg * 4 + (J)]; \
                xb = xo0[4u * bstride4 + hg * 4 + (J)]; \
                qd4 = fma(lo4, xa, qd4); \
                qd4 = fma(hi4, xb, qd4); \
            } \
            if (r5) { \
                xa = xe0[5u * bstride4 + hg * 4 + (J)]; \
                xb = xo0[5u * bstride4 + hg * 4 + (J)]; \
                qd5 = fma(lo4, xa, qd5); \
                qd5 = fma(hi4, xb, qd5); \
            } \
            if (r6) { \
                xa = xe0[6u * bstride4 + hg * 4 + (J)]; \
                xb = xo0[6u * bstride4 + hg * 4 + (J)]; \
                qd6 = fma(lo4, xa, qd6); \
                qd6 = fma(hi4, xb, qd6); \
            } \
            if (r7) { \
                xa = xe0[7u * bstride4 + hg * 4 + (J)]; \
                xb = xo0[7u * bstride4 + hg * 4 + (J)]; \
                qd7 = fma(lo4, xa, qd7); \
                qd7 = fma(hi4, xb, qd7); \
            }
        QMN_WORD(w4.x, 0) QMN_WORD(w4.y, 1)
        QMN_WORD(w4.z, 2) QMN_WORD(w4.w, 3)
        #undef QMN_WORD
        float qdf0 = (float)qd0.x + (float)qd0.y + (float)qd0.z + (float)qd0.w;
        acc0 = fma(s, qdf0, fma(b, xsum[hg], acc0));
        if (r1) {
            float qdf1 = (float)qd1.x + (float)qd1.y + (float)qd1.z + (float)qd1.w;
            acc1 = fma(s, qdf1, fma(b, xsum[1u * halves_per_row + hg], acc1));
        }
        if (r2) {
            float qdf2 = (float)qd2.x + (float)qd2.y + (float)qd2.z + (float)qd2.w;
            acc2 = fma(s, qdf2, fma(b, xsum[2u * halves_per_row + hg], acc2));
        }
        if (r3) {
            float qdf3 = (float)qd3.x + (float)qd3.y + (float)qd3.z + (float)qd3.w;
            acc3 = fma(s, qdf3, fma(b, xsum[3u * halves_per_row + hg], acc3));
        }
        if (r4) {
            float qdf4 = (float)qd4.x + (float)qd4.y + (float)qd4.z + (float)qd4.w;
            acc4 = fma(s, qdf4, fma(b, xsum[4u * halves_per_row + hg], acc4));
        }
        if (r5) {
            float qdf5 = (float)qd5.x + (float)qd5.y + (float)qd5.z + (float)qd5.w;
            acc5 = fma(s, qdf5, fma(b, xsum[5u * halves_per_row + hg], acc5));
        }
        if (r6) {
            float qdf6 = (float)qd6.x + (float)qd6.y + (float)qd6.z + (float)qd6.w;
            acc6 = fma(s, qdf6, fma(b, xsum[6u * halves_per_row + hg], acc6));
        }
        if (r7) {
            float qdf7 = (float)qd7.x + (float)qd7.y + (float)qd7.z + (float)qd7.w;
            acc7 = fma(s, qdf7, fma(b, xsum[7u * halves_per_row + hg], acc7));
        }
    }
    acc0 = simd_sum(acc0);
    if (lane == 0) y[row] = acc0;
    if (r1) {
        acc1 = simd_sum(acc1);
        if (lane == 0) y[1u * (ulong)p.out_dim + row] = acc1;
    }
    if (r2) {
        acc2 = simd_sum(acc2);
        if (lane == 0) y[2u * (ulong)p.out_dim + row] = acc2;
    }
    if (r3) {
        acc3 = simd_sum(acc3);
        if (lane == 0) y[3u * (ulong)p.out_dim + row] = acc3;
    }
    if (r4) {
        acc4 = simd_sum(acc4);
        if (lane == 0) y[4u * (ulong)p.out_dim + row] = acc4;
    }
    if (r5) {
        acc5 = simd_sum(acc5);
        if (lane == 0) y[5u * (ulong)p.out_dim + row] = acc5;
    }
    if (r6) {
        acc6 = simd_sum(acc6);
        if (lane == 0) y[6u * (ulong)p.out_dim + row] = acc6;
    }
    if (r7) {
        acc7 = simd_sum(acc7);
        if (lane == 0) y[7u * (ulong)p.out_dim + row] = acc7;
    }
}
