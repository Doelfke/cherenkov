// MTP input folding into the hyper-connection streams.

// hyper[b][g][i] = fe[b][i] + fh[b*groups+g][i]  (MTP input fold)
kernel void fn_mtp_fold(
    device const float* fe    [[buffer(0)]],
    device const float* fh    [[buffer(1)]],
    device float*       hyper [[buffer(2)]],
    constant GroupParams& p   [[buffer(3)]],
    constant uint&      nb    [[buffer(4)]],
    uint gi [[thread_position_in_grid]])
{
    const uint hh = p.n * p.groups;
    if (gi >= nb * hh) return;
    const uint b = gi / hh;
    const uint rem = gi % hh;
    const uint g = rem / p.n;
    const uint i = rem % p.n;
    hyper[gi] = fe[(ulong)b * p.n + i] + fh[((ulong)b * p.groups + g) * p.n + i];
}
