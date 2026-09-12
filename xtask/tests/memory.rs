use anyhow::Result;
use serde_json::json;
use xtask::metrics;

const TELEMETRY: &str = include_str!("fixtures/telemetry.txt");

#[test]
fn structured_memory_preserves_integers_and_overrides_rounded_text() -> Result<()> {
    let snapshot = json!({
        "metal_allocated_bytes_observed": 20_984_123_457_u64,
        "mapped_weight_buffer_bytes": 2_930_278_400_u64,
        "kv_index_capacity_bytes": 184_031_744,
        "context_capacity_tokens": 8192,
        "mtp_enabled": true,
        "expert_pool_bytes": 15_000_000_000_u64,
        "expert_pool_slots": 6000,
        "resident_experts": 5000,
        "future_integer": 9_007_199_254_740_993_u64
    });
    let parsed = metrics::parse(&format!("{TELEMETRY}\nmemory_stats {snapshot}\n"))?;

    assert_eq!(parsed["memory"], snapshot);
    assert_eq!(parsed["metal_gb"], 20.984123457);

    Ok(())
}

#[test]
fn legacy_telemetry_has_no_invented_memory_breakdown() -> Result<()> {
    let parsed = metrics::parse(TELEMETRY)?;

    assert!(parsed["memory"].is_null());
    assert_eq!(parsed["metal_gb"], 20.98);

    Ok(())
}

#[test]
fn malformed_or_ambiguous_snapshots_are_errors() {
    for line in [
        "{",
        "null",
        "[]",
        "{}",
        r#"{"metal_allocated_bytes_observed":-1}"#,
        r#"{"metal_allocated_bytes_observed":1.5}"#,
        r#"{"metal_allocated_bytes_observed":"12"}"#,
    ] {
        assert!(metrics::parse(&format!("{TELEMETRY}\nmemory_stats {line}\n")).is_err());
    }

    let line = "memory_stats {\"metal_allocated_bytes_observed\":12}\n";

    assert!(metrics::parse(&format!("{TELEMETRY}\n{line}{line}")).is_err());
}

#[test]
fn gpu_span_accepts_old_and_corrected_labels() -> Result<()> {
    for telemetry in [
        TELEMETRY.to_owned(),
        TELEMETRY.replace("gpu-active", "gpu-span"),
    ] {
        let metrics = metrics::parse(&telemetry)?;

        assert_eq!(metrics["gpu_span_ms"], 200.0);
        assert!(metrics.get("gpu_active_ms").is_none());
    }

    Ok(())
}
