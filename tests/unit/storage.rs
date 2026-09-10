use super::*;

#[test]
fn portable_root_keeps_durable_data_separate_from_scratch_and_config() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::new(Some(dir.path())).unwrap();

    assert_eq!(paths.data, dir.path());
    assert_eq!(paths.scratch, dir.path().join("scratch"));
    assert_eq!(paths.config, dir.path().join("cherenkov.toml"));
    assert_ne!(paths.downloads(), paths.scratch);
    assert!(!paths.default_model().exists());
}

#[test]
fn defaults_pin_model_identity() {
    let paths = Paths::new(Some(Path::new("/models"))).unwrap();

    assert_eq!(
        paths.default_model(),
        paths.model(DEFAULT_REPO, DEFAULT_REVISION).unwrap()
    );
    assert_ne!(
        paths.default_model(),
        paths.model(DEFAULT_REPO, &"a".repeat(40)).unwrap()
    );
}

#[test]
fn xdg_defaults_ignore_empty_and_relative_overrides() {
    let home = Some(Path::new("/users/test"));

    for suffix in [".local/share", ".cache", ".config"] {
        for value in [None, Some(OsString::new()), Some("relative/path".into())] {
            assert_eq!(
                xdg_dir(value, home, suffix).unwrap(),
                Path::new("/users/test").join(suffix)
            );
        }
    }
}

#[test]
fn xdg_absolute_override_does_not_require_a_home_directory() {
    assert_eq!(
        xdg_dir(Some("/custom/base".into()), None, ".cache").unwrap(),
        Path::new("/custom/base")
    );
}

#[test]
fn xdg_fallback_requires_a_home_directory() {
    for value in [None, Some(OsString::new()), Some("relative/path".into())] {
        assert!(xdg_dir(value, None, ".cache").is_err());
    }
}

#[test]
fn model_paths_reject_traversal_and_unresolved_revisions() {
    let paths = Paths::new(Some(Path::new("/models"))).unwrap();

    for repo in [
        "../model",
        "owner/..",
        "owner/name/extra",
        "/absolute",
        "owner/*",
    ] {
        assert!(paths.model(repo, DEFAULT_REVISION).is_err());
    }

    for revision in ["main", "../escape", "", "1234"] {
        assert!(paths.model(DEFAULT_REPO, revision).is_err());
    }
}

#[test]
fn disk_budget_rejects_overflow_and_reports_unavailable_paths() {
    let dir = tempfile::tempdir().unwrap();

    assert!(require_space(dir.path(), u64::MAX).is_err());
    assert!(require_space(&dir.path().join("missing"), 0).is_err());
}
