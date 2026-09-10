use super::*;

/// Reference quantizer matching the documented MLX affine scheme, used to
/// build synthetic layers for round-trip tests.
fn quantize(rows: &[Vec<f32>]) -> (Vec<u32>, Vec<bf16>, Vec<bf16>) {
    let in_dim = rows[0].len();
    let mut words = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for row in rows {
        for group in row.as_chunks::<GROUP_SIZE>().0 {
            let max = group.iter().cloned().fold(f32::MIN, f32::max);
            let min = group.iter().cloned().fold(f32::MAX, f32::min);
            let scale = ((max - min) / 15.0).max(1e-8);
            let bias = min;
            scales.push(bf16::from_f32(scale));
            biases.push(bf16::from_f32(bias));
            let s = bf16::from_f32(scale).to_f32();
            let b = bf16::from_f32(bias).to_f32();
            for chunk in group.as_chunks::<NIBBLES_PER_WORD>().0 {
                let mut word = 0u32;
                for (j, &v) in chunk.iter().enumerate() {
                    let q = (((v - b) / s).round().clamp(0.0, 15.0)) as u32;
                    word |= q << (4 * j);
                }
                words.push(word);
            }
        }
    }
    assert_eq!(words.len(), rows.len() * in_dim / NIBBLES_PER_WORD);
    (words, scales, biases)
}

#[test]
fn matvec_matches_dequant_reference() {
    let out_dim = 8;
    let in_dim = 128;
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|r| {
            (0..in_dim)
                .map(|i| ((r * 31 + i * 7) % 97) as f32 / 43.0 - 1.0)
                .collect()
        })
        .collect();
    let (words, scales, biases) = quantize(&rows);
    let q = QLinear {
        out_dim,
        in_dim,
        weight: &words,
        scales: &scales,
        biases: &biases,
    };
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i * 13) % 29) as f32 / 17.0 - 0.8)
        .collect();

    // Reference: dequantize rows then dense dot.
    let mut dense = vec![0.0f32; in_dim];
    let mut y_ref = vec![0.0f32; out_dim];
    for r in 0..out_dim {
        q.dequant_row(r, &mut dense);
        // Row must round-trip close to the original (quantization error only).
        for i in 0..in_dim {
            assert!((dense[i] - rows[r][i]).abs() < 0.2, "row {r} elem {i}");
        }
        y_ref[r] = dense.iter().zip(&x).map(|(a, b)| a * b).sum();
    }

    let mut y = vec![0.0f32; out_dim];
    q.matvec(&x, &mut y);
    for r in 0..out_dim {
        assert!(
            (y[r] - y_ref[r]).abs() < 1e-3 * y_ref[r].abs().max(1.0),
            "row {r}: {} vs {}",
            y[r],
            y_ref[r]
        );
    }
}
