use super::*;

#[test]
fn index_selects_unique_weight_files_and_rejects_unsafe_paths() {
    let files = shards(br#"{"weight_map":{"a":"model-1.safetensors","b":"model-1.safetensors","c":"model-2.safetensors"}}"#).unwrap();

    assert_eq!(files.len(), 2);

    for name in [
        "../secret.safetensors",
        "/weights.safetensors",
        "*.safetensors",
        "model.bin",
    ] {
        let bytes = serde_json::to_vec(&serde_json::json!({"weight_map":{"a":name}})).unwrap();

        assert!(shards(&bytes).is_err());
    }

    assert!(shards(br#"{"weight_map":{}}"#).is_err());
}

#[test]
fn cache_accounting_requires_complete_files_at_expected_sizes() {
    let dir = tempfile::tempdir().unwrap();
    let files = BTreeSet::from(["a".into(), "b".into()]);
    let available = BTreeMap::from([("a".into(), 5), ("b".into(), 7)]);

    assert_eq!(missing_bytes(&files, &available, dir.path()).unwrap(), 12);
    std::fs::write(dir.path().join("a"), b"12345").unwrap();
    std::fs::write(dir.path().join("b"), b"12").unwrap();
    assert_eq!(missing_bytes(&files, &available, dir.path()).unwrap(), 7);
    assert!(missing_bytes(&BTreeSet::from(["unknown".into()]), &available, dir.path()).is_err());
}

#[test]
fn publishing_reuses_blob_links_and_preserves_existing_stores() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("snapshot");
    let model = dir.path().join("model");

    std::fs::create_dir(&source).unwrap();
    std::fs::write(source.join("config.json"), b"{}").unwrap();

    let files = BTreeSet::from(["config.json".into()]);

    publish(&model, &source, &files).unwrap();

    let packed = model.join("packed");

    std::fs::create_dir(&packed).unwrap();
    std::fs::write(packed.join("experts2.bin"), b"two").unwrap();
    std::fs::write(packed.join("experts3.bin"), b"three").unwrap();
    publish(&model, &source, &files).unwrap();
    assert!(model.join("config.json").is_symlink());
    assert_eq!(std::fs::read(packed.join("experts2.bin")).unwrap(), b"two");
    assert_eq!(
        std::fs::read(packed.join("experts3.bin")).unwrap(),
        b"three"
    );
    std::fs::remove_file(model.join("config.json")).unwrap();
    std::fs::write(model.join("config.json"), b"local").unwrap();
    assert!(publish(&model, &source, &files).is_err());
    assert_eq!(std::fs::read(model.join("config.json")).unwrap(), b"local");
}
