//! The GEMM projection against the CPU matvec on real weights
//! (needs the packed model; skipped when it is absent).

use super::*;

fn model_dir() -> Option<std::path::PathBuf> {
    crate::storage::test_model_dir()
}

#[test]
fn qmm_matches_matvec() {
    let Some(dir) = model_dir() else { return };
    let packed = Packed::open(&dir).unwrap();
    let gpu = Gpu::load(
        &packed,
        64,
        &Options {
            pool_gb: PoolBudget::Gb(1.0),
            ..Options::default()
        },
    )
    .unwrap();
    let m = "language_model.model.layers.0";
    let cases = [
        (
            format!("{m}.attn_hyper_connection.input_mix_weight_down"),
            1usize,
        ),
        (
            format!("{m}.attn_hyper_connection.input_mix_weight_down"),
            19,
        ),
        (format!("{m}.attn_hyper_connection.input_mix_weight_up"), 3),
        (format!("{m}.attn_hyper_connection.block_inject_weight"), 2),
        (format!("{m}.linear_attn.in_proj_a"), 5),
        (format!("{m}.linear_attn.in_proj_qkv"), 40),
        (
            "language_model.model.layers.3.self_attn.q_proj".to_string(),
            33,
        ),
    ];
    let mut seed = 0x1234_5678u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    for (name, nb) in cases {
        let ql = packed.qlinear(&name).unwrap();
        let w = packed.manifest.dense(&format!("{name}.weight")).unwrap();
        let s = packed.manifest.dense(&format!("{name}.scales")).unwrap();
        let b = packed.manifest.dense(&format!("{name}.biases")).unwrap();
        let q = Q {
            w: w.offset as usize,
            s: s.offset as usize,
            b: b.offset as usize,
            out: ql.out_dim as u32,
            inp: ql.in_dim as u32,
        };
        let rp = nb.div_ceil(32) * 32 + 32;
        let x: Vec<f32> = (0..rp * ql.in_dim).map(|_| rnd()).collect();
        let xb = gpu.ctx.new_buffer(x.len() * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), xb.contents().cast::<f32>().as_ptr(), x.len())
        };
        let yb = gpu.ctx.new_buffer(rp * ql.out_dim * 4).unwrap();
        let cb = gpu.ctx.queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        gpu.qmm(&enc, &q, &xb, &yb, nb);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let y = unsafe {
            std::slice::from_raw_parts(yb.contents().cast::<f32>().as_ptr(), nb * ql.out_dim)
        };
        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        let mut r = vec![0.0f32; ql.out_dim];
        for row in 0..nb {
            ql.matvec(&x[row * ql.in_dim..(row + 1) * ql.in_dim], &mut r);
            for (a, c) in r.iter().zip(&y[row * ql.out_dim..(row + 1) * ql.out_dim]) {
                worst = worst.max((a - c).abs());
                scale = scale.max(a.abs());
            }
        }
        eprintln!("{name} nb={nb}: max abs diff {worst:.4} (scale {scale:.2})");
        assert!(
            worst <= 0.02 * scale.max(1.0),
            "{name} nb={nb}: qmm differs from matvec ({worst} vs scale {scale})"
        );
    }
}

#[test]
fn low_bit_expert_gemms_match_reconstructed_weights() {
    use std::os::unix::fs::FileExt;
    let Some(dir) = model_dir() else { return };
    let packed = Packed::open(&dir).unwrap();
    let gpu = Gpu::load(
        &packed,
        64,
        &Options {
            pool_gb: PoolBudget::Gb(0.25),
            ..Options::default()
        },
    )
    .unwrap();
    // Use actual cached records, but derive the oracle from their original
    // Q4 codes, independently of the compressed layout and GPU unpacking.
    let (layer, expert) = (3, 17);
    let original = packed.expert(layer, expert);
    for bits in [2, 3] {
        let path = packed.dir.join(format!("experts{bits}.bin"));
        let Ok(file) = std::fs::File::open(&path) else {
            eprintln!("skipping Q{bits} GEMMs: {} is absent", path.display());
            continue;
        };
        let l = crate::qwen4_exp::lowbit::Layout::new(&packed.manifest.experts, bits).unwrap();
        let mut record = vec![0u8; l.stride];
        let rid = layer * packed.manifest.experts.experts + expert;
        file.read_exact_at(&mut record, (rid * l.stride) as u64)
            .unwrap();
        let wb = gpu.ctx.new_buffer(l.stride).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                record.as_ptr(),
                wb.contents().cast::<u8>().as_ptr(),
                record.len(),
            );
        }
        for (name, ql, w, s, b) in [
            ("gate", &original.gate, l.gate_w, l.gate_s, l.gate_b),
            ("up", &original.up, l.up_w, l.up_s, l.up_b),
            ("down", &original.down, l.down_w, l.down_s, l.down_b),
        ] {
            let q = Q {
                w,
                s,
                b,
                out: ql.out_dim as u32,
                inp: ql.in_dim as u32,
            };
            let padded = 96;
            let x: Vec<f32> = (0..padded * ql.in_dim)
                .map(|i| (((i * 37 + i / ql.in_dim * 19) % 257) as f32 - 128.0) / 129.0)
                .collect();
            let xb = gpu.ctx.new_buffer(x.len() * 4).unwrap();
            let yb = gpu.ctx.new_buffer(padded * ql.out_dim * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr(),
                    xb.contents().cast::<f32>().as_ptr(),
                    x.len(),
                );
            }
            // Cover output fragment boundaries and both ends of each matrix.
            let channels = [0, 1, 7, 8, 31, 32, 63, 64, 127, 128, ql.out_dim - 1];
            let references: Vec<Vec<f32>> = channels
                .iter()
                .map(|&out| {
                    let weights: Vec<f32> = (0..ql.in_dim)
                        .map(|k| {
                            let code =
                                (ql.weight[out * ql.in_dim / 8 + k / 8] >> (4 * (k % 8))) & 15;
                            let step = 1u32 << (4 - bits);
                            let midpoint = (code / step * step) as f32 + (step - 1) as f32 * 0.5;
                            let group = out * ql.in_dim / 64 + k / 64;
                            ql.scales[group]
                                .to_f32()
                                .mul_add(midpoint, ql.biases[group].to_f32())
                        })
                        .collect();
                    x.chunks_exact(ql.in_dim)
                        .take(65)
                        .map(|row| {
                            row.iter()
                                .zip(&weights)
                                .map(|(&x, &w)| x as f64 * w as f64)
                                .sum::<f64>() as f32
                        })
                        .collect()
                })
                .collect();
            for nb in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65] {
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        yb.contents().cast::<f32>().as_ptr(),
                        padded * ql.out_dim,
                    )
                };
                output.fill(f32::NAN);
                let cb = gpu.ctx.queue.commandBuffer().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                gpu.expert_qmm_from(&enc, &wb, &q, &xb, &yb, nb, bits);
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                let mut worst = 0.0f32;
                let scale = references
                    .iter()
                    .flatten()
                    .fold(0.0f32, |a, &b| a.max(b.abs()));
                for (&channel, reference) in channels.iter().zip(&references) {
                    for row in 0..nb {
                        let got = output[row * ql.out_dim + channel];
                        assert!(got.is_finite(), "Q{bits} {name} nb={nb} unwritten output");
                        worst = worst.max((got - reference[row]).abs());
                    }
                }
                let tile = if nb <= 8 {
                    8
                } else if nb <= 16 {
                    16
                } else {
                    32
                };
                assert!(
                    output[nb.div_ceil(tile) * tile * ql.out_dim..]
                        .iter()
                        .all(|v| v.is_nan()),
                    "write beyond token tile"
                );
                assert!(
                    worst <= 0.002 * scale.max(1.0),
                    "Q{bits} {name} nb={nb}: error {worst}, scale {scale}"
                );
                eprintln!("Q{bits} {name} nb={nb}: max error {worst:.6}, scale {scale:.3}");
            }
        }
    }
}

#[test]
fn prefill_preserves_expert_precision_through_decode() {
    let Some(dir) = model_dir() else { return };
    let packed = Packed::open(&dir).unwrap();
    for (bits, miss_bits) in [(4, 4), (3, 3), (2, 2), (4, 2)] {
        if miss_bits < 4 && !packed.dir.join(format!("experts{miss_bits}.bin")).exists() {
            continue;
        }
        let options = Options {
            experts: bits,
            miss_experts: Some(miss_bits),
            // Fit a complete decode step while still forcing ring-slot reuse.
            pool_gb: PoolBudget::Gb(4.0),
            ..Options::default()
        };
        let mut gpu = Gpu::load(&packed, 128, &options).unwrap();
        let tokens: Vec<u32> = (128..192).collect();
        gpu.prefill_chunk(&tokens[..32], Some(tokens[32]), false)
            .unwrap();
        let (next, draft) = gpu.prefill_chunk(&tokens[32..], None, false).unwrap();
        assert_eq!(gpu.pos, 64);
        assert_eq!(gpu.mtp_len, 64);
        assert_eq!(gpu.tokens, tokens);
        assert_prefill_fetch_precision(&gpu, bits, miss_bits);
        let resident: Vec<usize> = (0..packed.manifest.experts.layers
            * packed.manifest.experts.experts)
            .filter(|&rid| gpu.res.is_member(rid))
            .collect();
        assert!(!resident.is_empty());
        let expected_kind = if bits < 4 {
            gpu.low_bit_store.unwrap().kind()
        } else {
            0
        };
        assert!(
            resident
                .iter()
                .all(|&rid| gpu.res.kind(rid) == expected_kind)
        );
        gpu.prefill_release();
        let verified = gpu.step_rows(&[next, draft], true, true).unwrap();
        assert!(
            verified
                .iter()
                .all(|&token| (token as usize) < packed.cfg.vocab_size)
        );
        assert!(gpu.logits_row(0).iter().all(|x| x.is_finite()));
        gpu.commit(1).unwrap();
        assert_eq!(gpu.pos, 65);
    }
}

fn assert_prefill_fetch_precision(gpu: &Gpu<'_>, bits: u32, miss_bits: u32) {
    let stride4 = gpu.p.manifest.experts.record_stride as usize;
    for st in &gpu.prefill_stats {
        assert!(st.fetched > RING);
        if bits == 4 && miss_bits == 4 {
            assert_eq!(st.fetched_bytes, st.fetched * stride4);
        } else if bits < 4 {
            let layout = gpu.low_bit_store.unwrap();
            assert_eq!(st.fetched_bytes, st.fetched * layout.stride);
        } else {
            let stride_low = gpu.low_bit_store.unwrap().stride;
            assert!(st.fetched_bytes < st.fetched * stride4);
            assert!(st.fetched_bytes > st.fetched * stride_low);
        }
    }
}
