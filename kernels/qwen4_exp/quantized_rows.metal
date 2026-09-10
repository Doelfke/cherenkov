// Q4 row helpers shared by hyper-connections and experts; compiled before both.

// Affine-Q4 dot of one weight row against NB half-stream rows (the
// qmv_multi_h inner loop). Row r reads xe/xo + r*bstride4 (half4 units)
// and xsum + r*gstride. Leaves the simd-reduced sums in acc[].
template <uint NB>
static inline void fn_q4_rows_h(
    device const uint4*  wr,
    device const bfloat* sr,
    device const bfloat* br,
    device const half4*  xe,
    device const half4*  xo,
    device const float*  xsum,
    uint halves_per_row,
    uint bstride4,
    uint gstride,
    uint lane,
    thread float* acc)
{
    for (uint hg = lane; hg < halves_per_row; hg += 32) {
        const uint4 w4 = wr[hg];
        const float s = (float)sr[hg >> 1];
        const float bb = (float)br[hg >> 1];
        half4 qd[NB];
        for (uint r = 0; r < NB; r++) qd[r] = 0.0h;
        uint word;
        half4 lo4;
        half4 hi4;
        #define FN_ROWS_WORD(W, J)                                                  \
            word = (W);                                                             \
            lo4 = half4(as_type<uchar4>(word & 0x0F0F0F0Fu));                       \
            hi4 = half4(as_type<uchar4>((word >> 4) & 0x0F0F0F0Fu));                \
            for (uint r = 0; r < NB; r++) {                                         \
                qd[r] = fma(lo4, xe[r * bstride4 + hg * 4 + (J)], qd[r]);           \
                qd[r] = fma(hi4, xo[r * bstride4 + hg * 4 + (J)], qd[r]);           \
            }
        FN_ROWS_WORD(w4.x, 0) FN_ROWS_WORD(w4.y, 1)
        FN_ROWS_WORD(w4.z, 2) FN_ROWS_WORD(w4.w, 3)
        #undef FN_ROWS_WORD
        for (uint r = 0; r < NB; r++) {
            const float qdf = (float)qd[r].x + (float)qd[r].y + (float)qd[r].z + (float)qd[r].w;
            acc[r] = fma(s, qdf, fma(bb, xsum[r * gstride + hg], acc[r]));
        }
    }
    for (uint r = 0; r < NB; r++) acc[r] = simd_sum(acc[r]);
}

// Affine-Q4 dot of TWO weight rows against NB half-stream rows, sharing
// every x load. Same per-row lane assignment and reduction order as
// fn_q4_rows_h, so results are bit-identical; the point is that a short
// row (the experts' 640-wide down projection is 20 half-groups over 32
// lanes, one iteration each) amortizes its setup and x loads over two
// output rows. Measured 26 percent faster at that shape.
template <uint NB>
static inline void fn_q4_2rows_h(
    device const uint4*  wr0,
    device const uint4*  wr1,
    device const bfloat* sr0,
    device const bfloat* br0,
    device const bfloat* sr1,
    device const bfloat* br1,
    device const half4*  xe,
    device const half4*  xo,
    device const float*  xsum,
    uint halves_per_row,
    uint bstride4,
    uint gstride,
    uint lane,
    thread float* acc0,
    thread float* acc1)
{
    for (uint hg = lane; hg < halves_per_row; hg += 32) {
        const uint4 w0 = wr0[hg];
        const uint4 w1 = wr1[hg];
        const float s0 = (float)sr0[hg >> 1];
        const float b0 = (float)br0[hg >> 1];
        const float s1 = (float)sr1[hg >> 1];
        const float b1 = (float)br1[hg >> 1];
        half4 qd0[NB];
        half4 qd1[NB];
        for (uint r = 0; r < NB; r++) { qd0[r] = 0.0h; qd1[r] = 0.0h; }
        #define FN_2ROWS_WORD(W0, W1, J)                                              \
        {                                                                             \
            const uint word0 = (W0);                                                  \
            const uint word1 = (W1);                                                  \
            const half4 lo0 = half4(as_type<uchar4>(word0 & 0x0F0F0F0Fu));            \
            const half4 hi0 = half4(as_type<uchar4>((word0 >> 4) & 0x0F0F0F0Fu));     \
            const half4 lo1 = half4(as_type<uchar4>(word1 & 0x0F0F0F0Fu));            \
            const half4 hi1 = half4(as_type<uchar4>((word1 >> 4) & 0x0F0F0F0Fu));     \
            for (uint r = 0; r < NB; r++) {                                           \
                const half4 xa = xe[r * bstride4 + hg * 4 + (J)];                     \
                const half4 xb = xo[r * bstride4 + hg * 4 + (J)];                     \
                qd0[r] = fma(lo0, xa, qd0[r]);                                        \
                qd0[r] = fma(hi0, xb, qd0[r]);                                        \
                qd1[r] = fma(lo1, xa, qd1[r]);                                        \
                qd1[r] = fma(hi1, xb, qd1[r]);                                        \
            }                                                                         \
        }
        FN_2ROWS_WORD(w0.x, w1.x, 0) FN_2ROWS_WORD(w0.y, w1.y, 1)
        FN_2ROWS_WORD(w0.z, w1.z, 2) FN_2ROWS_WORD(w0.w, w1.w, 3)
        #undef FN_2ROWS_WORD
        for (uint r = 0; r < NB; r++) {
            const float f0 = (float)qd0[r].x + (float)qd0[r].y + (float)qd0[r].z + (float)qd0[r].w;
            const float f1 = (float)qd1[r].x + (float)qd1[r].y + (float)qd1[r].z + (float)qd1[r].w;
            const float xs = xsum[r * gstride + hg];
            acc0[r] = fma(s0, f0, fma(b0, xs, acc0[r]));
            acc1[r] = fma(s1, f1, fma(b1, xs, acc1[r]));
        }
    }
    for (uint r = 0; r < NB; r++) {
        acc0[r] = simd_sum(acc0[r]);
        acc1[r] = simd_sum(acc1[r]);
    }
}
