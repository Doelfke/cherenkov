use super::*;
use crate::sampling::Sampling;

fn tokenizer() -> tok::ChatTokenizer {
    tok::ChatTokenizer {
        template: None,
        inner: tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
        im_end: 10,
        endoftext: 11,
    }
}

#[test]
fn suspended_decode_retains_pending_token_verifier_and_rng() {
    let tok = tokenizer();
    let options = Options {
        sampling: Sampling {
            temperature: 0.8,
            seed: Some(42),
            ..Sampling::default()
        },
        ..Options::default()
    };
    let logits = vec![0.0, 1.0, 2.0];
    let seed = PrefillResume {
        next: 2,
        drafts: vec![],
        logits: Some(logits.clone()),
    };
    let decoder = Decode::new(seed, &[0], &tok, &options, 8, None).unwrap();

    assert_eq!(decoder.n_draft, 0);

    let pending = decoder.cur;
    let rng = decoder.rng();
    let mut scheduler = std::collections::VecDeque::from([decoder]);
    // Unrelated generation must not consume this request's random stream.
    let mut other = Sampler::new(&options.sampling, &[1], 3).unwrap();

    for _ in 0..20 {
        other.sample(&logits).unwrap();
    }

    let resumed = scheduler.pop_front().unwrap();

    assert_eq!(resumed.cur, pending);
    assert_eq!(resumed.rng(), rng);

    let greedy = Decode::new(
        PrefillResume {
            next: 1,
            drafts: vec![2, 3],
            logits: None,
        },
        &[0],
        &tok,
        &Options::default(),
        8,
        None,
    )
    .unwrap();

    assert_eq!(greedy.drafts, [2, 3]);
    assert_eq!(greedy.n_draft, 2);
}

#[test]
fn continued_session_restores_rng_before_sampling_the_first_token() {
    let tok = tokenizer();
    let options = Options {
        sampling: Sampling {
            temperature: 0.7,
            seed: Some(12),
            ..Sampling::default()
        },
        ..Options::default()
    };
    let logits = vec![1.0, 1.5, 2.0];
    let mut reference = Sampler::new(&options.sampling, &[0], 3).unwrap();

    for _ in 0..17 {
        reference.sample(&logits).unwrap();
    }

    let checkpoint = reference.rng();
    let expected = reference.sample(&logits).unwrap();
    let resumed = Decode::new(
        PrefillResume {
            next: 2,
            drafts: vec![],
            logits: Some(logits),
        },
        &[0],
        &tok,
        &options,
        8,
        Some(checkpoint),
    )
    .unwrap();

    assert_eq!(resumed.cur, expected);
    assert_eq!(resumed.rng(), Some(reference.rng()));
}

#[test]
fn metal_handoff_restores_sampled_state_after_a_greedy_mtp_step() {
    let Some(dir) = crate::storage::test_model_dir() else {
        return;
    };
    let packed = qwen4_exp::packed::Packed::open(&dir).unwrap();
    let tok = tok::ChatTokenizer::load(&dir).unwrap();
    let greedy = Options {
        pool_gb: crate::options::PoolBudget::Gb(8.0),
        no_eos: true,
        ..Options::default()
    };
    let sampled = Options {
        sampling: Sampling {
            temperature: 0.7,
            top_k: 20,
            seed: Some(42),
            ..Sampling::default()
        },
        ..greedy.clone()
    };
    let mut gpu = qwen4_exp::gpu::Gpu::load(&packed, 64, &greedy).unwrap();
    let ids = tok.encode("Hello there").unwrap();
    let mut seed = prefill_row_batches(&mut gpu, None, None, &ids, 0, None).unwrap();
    seed.logits = Some(gpu.logits_row(gpu.last_logits_row()).to_vec());
    let mut first = Decode::new(seed, &ids, &tok, &sampled, 3, None).unwrap();

    first.step(&mut gpu, None, None, &mut |_| Ok(())).unwrap();

    let (pending, rng) = (first.cur, first.rng());
    let mut saved = None;

    gpu.save_into(&mut saved);
    gpu.reset_request();

    let other = tok.encode("The capital of France is").unwrap();
    let seed = prefill_row_batches(&mut gpu, None, None, &other, 2, None).unwrap();
    let mut second = Decode::new(seed, &other, &tok, &greedy, 5, None).unwrap();

    second.step(&mut gpu, None, None, &mut |_| Ok(())).unwrap();
    assert!(second.n_draft > 0);

    let second_saved = gpu.save_prefix();
    let (second_pending, second_count) = (second.cur, second.tokens.len());
    let second_drafts = second.drafts.clone();

    restore_checkpoint(&mut gpu, saved.as_ref().unwrap());
    assert_eq!(first.cur, pending);
    assert_eq!(first.rng(), rng);
    first.step(&mut gpu, None, None, &mut |_| Ok(())).unwrap();
    assert_eq!(first.tokens[1], pending);
    assert_eq!(first.tokens.len(), 2);
    assert_ne!(first.rng(), rng);
    // Reuse the snapshot allocation after advancing, including growing KV data.
    gpu.save_into(&mut saved);
    assert!(gpu.save_prefix() == saved.unwrap());

    let before_limit = first.rng();

    assert_output_limit(&mut gpu, &mut first, 3);
    restore_checkpoint(&mut gpu, &second_saved);
    assert_eq!(second.drafts, second_drafts);
    second.step(&mut gpu, None, None, &mut |_| Ok(())).unwrap();
    assert_eq!(second.tokens[second_count], second_pending);
    assert_eq!(first.rng(), before_limit);
}

fn restore_checkpoint(gpu: &mut qwen4_exp::gpu::Gpu<'_>, saved: &qwen4_exp::gpu::PrefixState) {
    gpu.reset_request();
    gpu.restore_prefix(saved).unwrap();
    assert!(gpu.save_prefix() == *saved);
}

fn assert_output_limit(gpu: &mut qwen4_exp::gpu::Gpu<'_>, decoder: &mut Decode, limit: usize) {
    let rng = decoder.rng();

    decoder.step(gpu, None, None, &mut |_| Ok(())).unwrap();
    assert_eq!(decoder.finish_reason, Some("length"));
    assert_eq!(decoder.tokens.len(), limit);
    assert_eq!(decoder.rng(), rng); // No unused draw past the output boundary.
}
