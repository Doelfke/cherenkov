//! Optional CPU-reference comparisons, research dumps and per-step traces.

use crate::qwen4_exp;
use anyhow::Result;

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn logits_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    (max_abs, max_abs / scale)
}

pub(super) fn dump_run(gpu: &qwen4_exp::gpu::Gpu<'_>) -> Result<()> {
    if let Ok(path) = std::env::var("CHERENKOV_DUMP_EXPERTS") {
        std::fs::write(&path, serde_json::to_vec(&gpu.expert_history)?)?;
        eprintln!(
            "expert history ({} steps) written to {path}",
            gpu.expert_history.len()
        );
    }
    if let Ok(path) = std::env::var("CHERENKOV_DUMP_STATES") {
        // [step][block] = (router input, top-k), row 0 only; f32 little
        // endian after a small JSON header line.
        use std::io::Write as _;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let steps = gpu.state_history.len();
        let blocks = gpu.state_history.first().map_or(0, |s| s.len());
        let hidden = gpu
            .state_history
            .first()
            .and_then(|s| s.first())
            .map_or(0, |b| b.0.len());
        let k = gpu
            .state_history
            .first()
            .and_then(|s| s.first())
            .map_or(0, |b| b.1.len());
        // Steps carry 48 or 49 blocks (the folded MTP block only runs on
        // drafting steps), so each step is prefixed with its count.
        writeln!(
            f,
            "{{\"format\":2,\"steps\":{steps},\"blocks\":{blocks},\"hidden\":{hidden},\"k\":{k}}}"
        )?;
        for step in &gpu.state_history {
            f.write_all(&(step.len() as u32).to_le_bytes())?;
            for (x, ids) in step {
                for v in x {
                    f.write_all(&v.to_le_bytes())?;
                }
                for id in ids {
                    f.write_all(&id.to_le_bytes())?;
                }
            }
        }
        eprintln!("router states ({steps} steps x {blocks} blocks x {hidden}) written to {path}");
    }
    if let Ok(path) = std::env::var("CHERENKOV_DUMP_ROUTES") {
        std::fs::write(&path, serde_json::to_vec(&gpu.route_history)?)?;
        eprintln!(
            "route history ({} steps) written to {path}",
            gpu.route_history.len()
        );
    }
    if let Ok(path) = std::env::var("CHERENKOV_DUMP_LA") {
        std::fs::write(&path, serde_json::to_vec(&gpu.la_log)?)?;
        eprintln!(
            "lookahead log ({} predictions) written to {path}",
            gpu.la_log.len()
        );
    }
    Ok(())
}

/// Compare the GPU's rows against the CPU reference, which forwards the
/// same tokens one at a time. With `next` (the token after each row), the
/// CPU also runs the MTP head per row and the last row's draft logits are
/// compared with the GPU's.
#[allow(clippy::too_many_arguments)]
pub(super) fn qwen4_exp_check_rows(
    m: &qwen4_exp::cpu::CpuModel<'_>,
    st: &mut qwen4_exp::cpu::State,
    gpu: &qwen4_exp::gpu::Gpu<'_>,
    rows: &[u32],
    next: Option<&[u32]>,
    pos0: usize,
    label: &str,
    prefill: bool,
) -> Result<()> {
    for (r, &t) in rows.iter().enumerate() {
        let ref_logits = m.forward_token(t, st)?;
        let g = if prefill {
            gpu.pf_logits_row(r)
        } else {
            gpu.logits_row(r)
        };
        let (abs, rel) = logits_diff(g, &ref_logits);
        let (ga, ra) = (argmax(g), argmax(&ref_logits));
        eprintln!(
            "  {label} {}: gpu argmax {ga} cpu argmax {ra} | max abs diff {abs:.4} (rel {rel:.2e}){}",
            pos0 + r,
            if ga == ra { "" } else { "  <-- MISMATCH" }
        );
        if let Some(next) = next {
            let hyper = st.last_hyper.clone();
            let (ml, _) = m.mtp_forward(next[r], &hyper, pos0 + r, st)?;
            if r + 1 == rows.len() {
                let g = gpu.mtp_logits_row(if prefill { 0 } else { r });
                let (abs, rel) = logits_diff(g, &ml);
                let (ga, ra) = (argmax(g), argmax(&ml));
                eprintln!(
                    "  {label} {} mtp: gpu draft {ga} cpu draft {ra} | max abs diff {abs:.4} (rel {rel:.2e}){}",
                    pos0 + r,
                    if ga == ra { "" } else { "  <-- MISMATCH" }
                );
            }
        }
    }
    Ok(())
}

/// Record prompt argmax rows and the final long-context attention selection.
pub(super) fn dump_prefill_chunk(
    gpu: &qwen4_exp::gpu::Gpu<'_>,
    engine: bool,
    pos: usize,
    rows: usize,
    prompt_tokens: usize,
    lines: &mut Vec<String>,
) {
    for row in 0..rows {
        let logits = if engine {
            gpu.pf_logits_row(row)
        } else {
            gpu.logits_row(row)
        };
        let token = argmax(logits);
        lines.push(format!(
            "{} {token} {:.4}",
            pos + row,
            logits[token as usize]
        ));
    }
    let cfg = &gpu.p.cfg;
    if pos + rows != prompt_tokens
        || prompt_tokens <= cfg.indexer_budget + cfg.indexer_compress_ratio
    {
        return;
    }
    if engine {
        let blocks = gpu.debug_engine_blocks(rows - 1);
        eprintln!(
            "engine last row: {} blocks selected, last 8 {:?}",
            blocks.len(),
            &blocks[blocks.len().saturating_sub(8)..]
        );
        return;
    }
    let vis = gpu.debug_row_vis(rows - 1);
    let blocks: Vec<u32> = vis
        .iter()
        .filter(|&&t| t % 4 == 0)
        .map(|&t| t / 4)
        .collect();
    eprintln!(
        "row path last row: {} visible tokens, last 8 {:?}; {} block starts, last 8 {:?}",
        vis.len(),
        &vis[vis.len().saturating_sub(8)..],
        blocks.len(),
        &blocks[blocks.len().saturating_sub(8)..]
    );
}

pub(super) fn dump_decode(
    gpu: &qwen4_exp::gpu::Gpu<'_>,
    ids: &[u32],
    out: &[u32],
    prefill_steps: usize,
) -> Result<()> {
    if let Ok(path) = std::env::var("CHERENKOV_DUMP_TOKENS") {
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({ "prompt": ids, "out": out }))?,
        )?;
        eprintln!("tokens written to {path}");
    }
    if std::env::var_os("CHERENKOV_TRACE").is_some() {
        for i in prefill_steps..gpu.step_ms.len() {
            eprintln!(
                "  step {i}: {} rows, {:.1} ms wall, {:.1} ms gpu, {:.1} ms io wait, {} sync fetches ({:.0} MB), lookahead hit {:.2} fetched {}",
                gpu.rows[i],
                gpu.step_ms[i],
                gpu.gpu_ms[i],
                gpu.io_ms[i],
                gpu.misses[i],
                gpu.miss_bytes[i] as f64 / 1e6,
                gpu.lookahead_hit[i],
                gpu.lookahead_issued[i]
            );
        }
    }
    Ok(())
}
