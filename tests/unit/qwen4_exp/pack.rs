use super::*;
use serde_json::json;

fn checkpoint(dir: &Path) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut tensor = |name: String, dtype: &str, shape: &[usize], width: usize| {
        let start = data.len();
        let len = shape.iter().product::<usize>() * width;
        data.extend((0..len).map(|i| (i * 37 + 11) as u8));
        header.insert(
            name,
            json!({
                "dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]
            }),
        );
    };
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let prefix = format!("language_model.model.layers.0.mlp.switch_mlp.{projection}");
        tensor(format!("{prefix}.weight"), "U32", &[2, 64, 8], 4);
        tensor(format!("{prefix}.scales"), "BF16", &[2, 64, 1], 2);
        tensor(format!("{prefix}.biases"), "BF16", &[2, 64, 1], 2);
    }
    let ple = "language_model.model.ple";
    let ngram = format!("{ple}.ngram_embedding.shards.0");
    tensor(format!("{ngram}.weight"), "U32", &[2, 8], 4);
    tensor(format!("{ngram}.scales"), "BF16", &[2, 1], 2);
    tensor(format!("{ngram}.biases"), "BF16", &[2, 1], 2);
    for name in [
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
        "layer_multipliers",
    ] {
        tensor(format!("{ple}.{name}"), "I64", &[1], 8);
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
    for name in ["config.json", "tokenizer.json"] {
        std::fs::write(dir.join(name), "{}").unwrap();
    }
}

#[test]
fn fresh_checkpoint_builds_base_and_both_targets_at_custom_output() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path());
    let out = dir.path().join("custom/nested/store");
    prepare(dir.path(), Some(&out), &[2, 3]).unwrap();
    for name in [
        "manifest.json",
        "experts.bin",
        "experts2.bin",
        "experts3.bin",
        "manifest2.json",
        "manifest3.json",
        "dense.bin",
        "ngram.bin",
        "config.json",
        "tokenizer.json",
    ] {
        assert!(out.join(name).is_file(), "{name}");
    }
    assert!(!dir.path().join("packed").exists());
    let base = std::fs::read(out.join("experts.bin")).unwrap();
    prepare(dir.path(), Some(&out), &[4, 2, 3]).unwrap();
    assert_eq!(base, std::fs::read(out.join("experts.bin")).unwrap());
}

#[test]
fn default_pack_can_be_extended_from_the_packed_directory() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path());
    prepare(dir.path(), None, &[4]).unwrap();
    let out = dir.path().join("packed");
    assert!(!out.join("experts2.bin").exists());
    assert!(!out.join("experts3.bin").exists());
    prepare(&out, None, &[3]).unwrap();
    assert!(out.join("experts3.bin").is_file());
    assert!(!out.join("experts2.bin").exists());
    assert!(!out.join("packed").exists());
    let absent = dir.path().join("another");
    assert!(prepare(&out, Some(&absent), &[2]).is_err());
    assert!(!absent.exists());
}

#[test]
fn output_from_another_checkpoint_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path());
    prepare(dir.path(), None, &[4]).unwrap();
    let other = tempfile::tempdir().unwrap();
    let out = dir.path().join("packed");
    let error = prepare(other.path(), Some(&out), &[2]).unwrap_err();
    assert!(error.to_string().contains("different source model"));
    assert!(!out.join("experts2.bin").exists());
}

#[test]
fn relocated_model_reuses_its_local_base_store() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    std::fs::create_dir(&source).unwrap();
    checkpoint(&source);
    prepare(&source, None, &[4]).unwrap();
    let relocated = dir.path().join("relocated");
    std::fs::rename(source, &relocated).unwrap();
    prepare(&relocated, None, &[2]).unwrap();
    assert!(relocated.join("packed/experts2.bin").is_file());
}

#[test]
fn invalid_selection_does_not_create_a_base_store() {
    let dir = tempfile::tempdir().unwrap();
    for targets in [vec![], vec![2, 1], vec![5]] {
        assert!(prepare(dir.path(), None, &targets).is_err());
    }
    assert!(!dir.path().join("packed").exists());
}
