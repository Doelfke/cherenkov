// Dependent-FMA clock probe used to expose GPU throttling.

kernel void clock_probe(device float* out [[buffer(0)]],
                        constant uint& iters [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    float x = 1.0f + (float)gid * 1e-9f;
    for (uint i = 0; i < iters; i++) {
        x = fma(x, 1.0000001f, 1e-7f);
        x = fma(x, 0.9999999f, -1e-7f);
    }
    out[gid] = x;
}
