use super::*;

#[test]
fn greedy_ties_and_penalties_are_explicit() {
    let mut s = Sampler::new(&Sampling::default(), &[], 3).unwrap();
    assert_eq!(s.sample(&[1.0, 1.0, 0.0]).unwrap(), 0);
    let options = Sampling {
        presence_penalty: 1.0,
        frequency_penalty: 0.5,
        ..Sampling::default()
    };
    let mut s = Sampler::new(&options, &[0, 0], 3).unwrap();
    assert_eq!(s.sample(&[2.0, 1.0, 0.0]).unwrap(), 1);
    s.accept(1).unwrap();
    assert_eq!(s.sample(&[2.0, 1.0, 0.1]).unwrap(), 2);
}

#[test]
fn top_k_and_nucleus_keep_the_boundary_token() {
    let options = Sampling {
        temperature: 1.0,
        top_k: 1,
        seed: Some(1),
        ..Sampling::default()
    };
    let mut s = Sampler::new(&options, &[], 3).unwrap();
    for _ in 0..50 {
        assert_eq!(s.sample(&[0.0, 5.0, 2.0]).unwrap(), 1);
    }
    assert_eq!(nucleus(&[(0, 0.5), (1, 0.3), (2, 0.2)], 0.6), (2, 0.8));
    let mut s = Sampler::new(
        &Sampling {
            top_k: 0,
            top_p: 0.1,
            ..options
        },
        &[],
        3,
    )
    .unwrap();
    for _ in 0..50 {
        assert_eq!(s.sample(&[0.0, 5.0, 2.0]).unwrap(), 1);
    }
}

#[test]
fn distribution_and_seed_are_independent_of_other_requests() {
    let options = Sampling {
        temperature: 1.0,
        seed: Some(42),
        ..Sampling::default()
    };
    let mut a = Sampler::new(&options, &[], 2).unwrap();
    let mut b = Sampler::new(&options, &[], 2).unwrap();
    let mut unrelated = Sampler::new(
        &Sampling {
            seed: Some(8),
            ..options
        },
        &[],
        2,
    )
    .unwrap();
    let logits = [0.25_f32.ln(), 0.75_f32.ln()];
    let mut counts = [0; 2];
    for _ in 0..20_000 {
        let token = a.sample(&logits).unwrap();
        unrelated.sample(&[0.0, 0.0]).unwrap();
        assert_eq!(token, b.sample(&logits).unwrap());
        counts[token as usize] += 1;
    }
    assert!((counts[0] as f64 / 20_000.0 - 0.25).abs() < 0.02);
}

#[test]
fn rejects_invalid_parameters_and_logits() {
    for value in [
        serde_json::json!({"top_p":0}),
        serde_json::json!({"temperature":-1}),
        serde_json::json!({"top_k":-1}),
        serde_json::json!({"seed":"wrong"}),
    ] {
        assert!(Sampling::from_request(&value, &Sampling::default()).is_err());
    }
    let mut s = Sampler::new(&Sampling::default(), &[], 2).unwrap();
    assert!(s.sample(&[f32::NAN, 0.0]).is_err());
    assert!(s.sample(&[f32::NEG_INFINITY; 2]).is_err());
    assert_eq!(s.sample(&[f32::NEG_INFINITY, 0.0]).unwrap(), 1);
}
