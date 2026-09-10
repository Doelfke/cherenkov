use super::*;

/// The gathering flash-decode kernel over an identity token list must
/// reproduce the contiguous kernel's partials bit for bit.
#[test]
fn selective_attention_matches_dense() {
    let (n_heads, n_kv, hd, t_len) = (24usize, 2usize, 256usize, 100usize);
    let kv_row = n_kv * hd;
    let max_blk = ATTN_MAX_WG;
    let n_wg = 4usize;
    let mut rng = Lcg(11);
    let q: Vec<f32> = (0..n_heads * 2 * hd).map(|_| rng.f()).collect();
    let scale_off = t_len * kv_row;
    let mut kc = vec![0u8; scale_off + t_len * (kv_row / 32) * 2];
    let mut vc = kc.clone();

    for i in 0..scale_off {
        kc[i] = (rng.f() * 100.0) as i8 as u8;
        vc[i] = (rng.f() * 100.0) as i8 as u8;
    }

    for i in 0..t_len * (kv_row / 32) {
        let s = half::f16::from_f32(0.01 + 0.02 * rng.f().abs())
            .to_bits()
            .to_le_bytes();

        kc[scale_off + 2 * i..scale_off + 2 * i + 2].copy_from_slice(&s);

        let s = half::f16::from_f32(0.01 + 0.02 * rng.f().abs())
            .to_bits()
            .to_le_bytes();

        vc[scale_off + 2 * i..scale_off + 2 * i + 2].copy_from_slice(&s);
    }

    let vis: Vec<u32> = (0..t_len as u32).collect();
    let ctx = MetalContext::new().unwrap();
    let flib = ctx.compile_library(FORWARD_MSL).unwrap();
    let blib = ctx.compile_library(BATCH_MSL).unwrap();
    let dense = ctx.pipeline(&flib, "attn_part2_q8").unwrap();
    let sel = ctx.pipeline(&blib, "fn_attn_part2_q8_sel").unwrap();
    let q_b = upload(&ctx, &q);
    let kc_b = upload(&ctx, &kc);
    let vc_b = upload(&ctx, &vc);
    let vis_b = upload(&ctx, &vis);
    let n_part = n_heads * max_blk * (2 + hd);
    let ap = AttnPartParams {
        n_heads: n_heads as u32,
        n_kv: n_kv as u32,
        head_dim: hd as u32,
        t_len: t_len as u32,
        q_stride: 2 * hd as u32,
        q_off: 0,
        max_blk: max_blk as u32,
        scale: (hd as f32).powf(-0.5) * std::f32::consts::LOG2_E,
    };
    let (n_wg_u, so) = (n_wg as u32, scale_off as u32);
    let mut outs = Vec::new();

    for (pso, with_vis) in [(&dense, false), (&sel, true)] {
        let part = upload(&ctx, &vec![0.0f32; n_part]);
        let cb = ctx.queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(pso);

        unsafe {
            enc.setBuffer_offset_atIndex(Some(&q_b), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&kc_b), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&vc_b), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&part), 0, 3);
            set_bytes(&enc, 4, &ap);
            set_bytes(&enc, 5, &n_wg_u);
            set_bytes(&enc, 6, &so);

            if with_vis {
                enc.setBuffer_offset_atIndex(Some(&vis_b), 0, 7);
            }
        }

        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_kv * n_wg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32 * (n_heads / n_kv),
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        outs.push(download::<f32>(&part, n_part));
    }

    assert!(
        outs[0].iter().any(|v| *v != 0.0),
        "dense kernel wrote nothing"
    );
    assert_eq!(outs[0].len(), outs[1].len());

    let diff = outs[0]
        .iter()
        .zip(&outs[1])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    assert_eq!(diff, 0, "partials differ in {diff} elements");
}

/// The gathering flash-decode kernel plus combine over a random
/// visible subset against a CPU attention over the same tokens.
#[test]
fn selective_attention_matches_reference() {
    selective_attention_case(300, 0.5);
    // The production shape: 512 selected blocks of a long context.
    selective_attention_case(2300, 512.0 / 575.0);
}

fn selective_attention_case(t_len: usize, keep: f32) {
    let (n_heads, n_kv, hd) = (24usize, 2usize, 256usize);
    let kv_row = n_kv * hd;
    let n_rep = n_heads / n_kv;
    let max_blk = ATTN_MAX_WG;
    let mut rng = Lcg(23);
    let q: Vec<f32> = (0..n_heads * 2 * hd).map(|_| rng.f()).collect();
    let scale_off = t_len * kv_row;
    let mut kc = vec![0u8; scale_off + t_len * (kv_row / 32) * 2];
    let mut vc = kc.clone();

    for i in 0..scale_off {
        kc[i] = (rng.f() * 100.0) as i8 as u8;
        vc[i] = (rng.f() * 100.0) as i8 as u8;
    }

    let mut ks = vec![0.0f32; t_len * kv_row / 32];
    let mut vs = ks.clone();

    for i in 0..t_len * (kv_row / 32) {
        ks[i] = 0.01 + 0.02 * rng.f().abs();
        vs[i] = 0.01 + 0.02 * rng.f().abs();

        kc[scale_off + 2 * i..scale_off + 2 * i + 2]
            .copy_from_slice(&half::f16::from_f32(ks[i]).to_bits().to_le_bytes());
        vc[scale_off + 2 * i..scale_off + 2 * i + 2]
            .copy_from_slice(&half::f16::from_f32(vs[i]).to_bits().to_le_bytes());
    }

    // Visible: every token of a random subset of the 4-token blocks, plus the tail.
    let mut vis: Vec<u32> = Vec::new();

    for b in 0..t_len / 4 {
        if (rng.f() + 1.0) * 0.5 < keep {
            vis.extend((b * 4..b * 4 + 4).map(|t| t as u32));
        }
    }

    vis.extend((t_len / 4 * 4..t_len).map(|t| t as u32));

    let n_vis = vis.len();
    // CPU reference with the kernels' dequantization and rounding of scales.
    let deq = |buf: &[u8], sc: &[f32], t: usize, e: usize| -> f32 {
        let s = half::f16::from_f32(sc[t * (kv_row / 32) + e / 32]).to_f32();

        (buf[t * kv_row + e] as i8) as f32 * s
    };
    let scale = (hd as f32).powf(-0.5);
    let mut expect = vec![0.0f32; n_heads * hd];

    for h in 0..n_heads {
        let hk = h / n_rep;
        let qh = &q[h * 2 * hd..h * 2 * hd + hd];
        let scores: Vec<f32> = vis
            .iter()
            .map(|&t| {
                (0..hd)
                    .map(|d| qh[d] * deq(&kc, &ks, t as usize, hk * hd + d))
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f32 = w.iter().sum();

        for d in 0..hd {
            let acc: f32 = vis
                .iter()
                .zip(&w)
                .map(|(&t, &wi)| wi * deq(&vc, &vs, t as usize, hk * hd + d))
                .sum();
            let gate = q[h * 2 * hd + hd + d];
            expect[h * hd + d] = acc / z / (1.0 + (-gate).exp());
        }
    }

    let ctx = MetalContext::new().unwrap();
    let flib = ctx.compile_library(FORWARD_MSL).unwrap();
    let blib = ctx.compile_library(BATCH_MSL).unwrap();
    let sel = ctx.pipeline(&blib, "fn_attn_part2_q8_sel").unwrap();
    let combine = ctx.pipeline(&flib, "attn_combine").unwrap();
    let q_b = upload(&ctx, &q);
    let kc_b = upload(&ctx, &kc);
    let vc_b = upload(&ctx, &vc);
    let vis_b = upload(&ctx, &vis);
    let n_wg = n_vis.div_ceil(32).div_ceil(3).clamp(1, ATTN_MAX_WG);
    let part = upload(&ctx, &vec![0.0f32; n_heads * max_blk * (2 + hd)]);
    let out = upload(&ctx, &vec![0.0f32; n_heads * hd]);
    let ap = AttnPartParams {
        n_heads: n_heads as u32,
        n_kv: n_kv as u32,
        head_dim: hd as u32,
        t_len: n_vis as u32,
        q_stride: 2 * hd as u32,
        q_off: 0,
        max_blk: max_blk as u32,
        scale: scale * std::f32::consts::LOG2_E,
    };
    let (n_wg_u, so, nblk, out_off) = (n_wg as u32, scale_off as u32, n_wg as u32, 0u32);
    let cb = ctx.queue.commandBuffer().unwrap();
    let enc = cb.computeCommandEncoder().unwrap();

    enc.setComputePipelineState(&sel);

    unsafe {
        enc.setBuffer_offset_atIndex(Some(&q_b), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&kc_b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&vc_b), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&part), 0, 3);
        set_bytes(&enc, 4, &ap);
        set_bytes(&enc, 5, &n_wg_u);
        set_bytes(&enc, 6, &so);
        enc.setBuffer_offset_atIndex(Some(&vis_b), 0, 7);
    }

    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_kv * n_wg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32 * n_rep,
            height: 1,
            depth: 1,
        },
    );
    enc.setComputePipelineState(&combine);

    unsafe {
        enc.setBuffer_offset_atIndex(Some(&q_b), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&part), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
        set_bytes(&enc, 3, &ap);
        set_bytes(&enc, 4, &nblk);
        set_bytes(&enc, 5, &out_off);
    }

    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let got: Vec<f32> = download(&out, n_heads * hd);
    let worst = got
        .iter()
        .zip(&expect)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale_v = expect.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    assert!(
        worst <= 2e-3 * scale_v.max(1e-3),
        "selective attention differs from reference: {worst} vs scale {scale_v} ({n_vis} visible of {t_len})"
    );
}
