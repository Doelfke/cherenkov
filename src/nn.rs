//! Small CPU math helpers for the f32 reference path.

use half::bf16;

/// RMSNorm: x_i * w_i / sqrt(mean(x^2) + eps). Computed in f32.
pub fn rms_norm(x: &mut [f32], w: &[bf16], eps: f32) {
    debug_assert_eq!(x.len(), w.len());
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for (v, wi) in x.iter_mut().zip(w) {
        *v = *v * inv * wi.to_f32();
    }
}

/// SiLU activation: x * sigmoid(x).
pub fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

pub fn softplus(v: f32) -> f32 {
    // Numerically stable log(1 + e^v).
    if v > 20.0 { v } else { v.exp().ln_1p() }
}

/// In-place softmax over `x`.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// L2-normalize with eps, in place: x / sqrt(sum(x^2) + eps).
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>();
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}
