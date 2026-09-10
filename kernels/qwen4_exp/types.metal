// Host ABI types shared by qwen4-exp row-batched subsystems.

struct FnQmvParams {
    uint out_dim;
    uint in_dim;
};

struct GroupParams {
    uint n;        // group width (hidden)
    uint groups;   // streams
    float eps;
    float shift;   // added to the norm weight (1.0 for raw HF norms)
};
