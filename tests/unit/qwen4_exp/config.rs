use super::*;

#[test]
fn configuration_rejects_other_architectures_before_loading_weights() {
    let dir = tempfile::tempdir().unwrap();
    for (json, message) in [
        (
            serde_json::json!({"model_type":"qwen3_next"}),
            "unsupported model_type",
        ),
        (
            serde_json::json!({"model_type":"qwen4_exp", "text_config":{"model_type":"other"}}),
            "expected qwen4_exp_text",
        ),
        (serde_json::json!({}), "model_type missing"),
    ] {
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&json).unwrap(),
        )
        .unwrap();
        let error = Qwen4ExpConfig::load(dir.path()).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
}
