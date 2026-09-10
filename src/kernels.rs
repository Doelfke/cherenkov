//! Metal translation units shared by the engine and GPU tests.
//!
//! Files are fragments, assembled at Rust compile time. Keep shared types
//! and helpers before their users; no filesystem includes are needed at
//! runtime. Each fragment gets its own filename/line numbers in diagnostics.

macro_rules! metal_library {
    ($($path:literal),+ $(,)?) => {
        concat!(
            "#include <metal_stdlib>\nusing namespace metal;\n",
            $(
                "\n#line 1 \"", $path, "\"\n",
                include_str!(concat!("../kernels/", $path)),
                "\n",
            )+
        )
    };
}

pub(crate) const FORWARD_MSL: &str = metal_library!(
    "common/quantized.metal",
    "common/gemm.metal",
    "common/attention.metal",
    "common/deltanet.metal",
    "common/elementwise.metal",
    "common/sampling.metal",
);

pub(crate) const BATCH_MSL: &str = metal_library!(
    "qwen4_exp/types.metal",
    // Q4 helpers serve both hyper-connections and experts. Q2/Q3 are
    // expert-only and stay with the expert kernels.
    "qwen4_exp/quantized_rows.metal",
    "qwen4_exp/hyperconnection.metal",
    "qwen4_exp/experts.metal",
    "qwen4_exp/expert_gemm.metal",
    "qwen4_exp/ple.metal",
    "qwen4_exp/mtp.metal",
    "qwen4_exp/deltanet.metal",
    "qwen4_exp/qsa.metal",
    "qwen4_exp/rows.metal",
);

/// Device diagnostic, compiled separately from model kernels.
pub(crate) const CLOCK_PROBE_MSL: &str = metal_library!("device/clock.metal");
