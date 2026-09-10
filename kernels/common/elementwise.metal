// Elementwise activation, residual addition, and buffer copying.

// y[i] = silu(a[i]) * b[i]
kernel void silu_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float*       y [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    float v = a[i];
    y[i] = (v / (1.0f + exp(-v))) * b[i];
}

// x[i] += r[i]
kernel void add_inplace(
    device float*       x [[buffer(0)]],
    device const float* r [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    x[i] += r[i];
}

// Plain device-to-device f32 copy.
kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) dst[i] = src[i];
}
