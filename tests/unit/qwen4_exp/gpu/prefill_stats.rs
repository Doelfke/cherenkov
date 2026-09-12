use super::*;

#[test]
fn aggregates_actual_chunk_sizes_and_timing() {
    let mut stats = PrefillStats::default();

    for tokens in [92, 92, 91, 1024] {
        stats.record(ChunkStats {
            tokens,
            secs: 2.0,
            wait_s: 0.25,
            gpu_experts_s: 1.0,
            ..Default::default()
        });
    }

    assert_eq!(stats.chunks, 4);
    assert_eq!(stats.tokens, 1299);
    assert_eq!(stats.min_chunk_tokens, 91);
    assert_eq!(stats.max_chunk_tokens, 1024);
    assert_eq!(stats.seconds, 8.0);
    assert_eq!(stats.ring_wait_seconds, 1.0);
    assert_eq!(stats.gpu_expert_seconds, 4.0);

    let json = serde_json::to_string(&stats).unwrap();
    let restored: PrefillStats = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.tokens, stats.tokens);
}
