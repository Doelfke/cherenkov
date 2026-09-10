use super::*;

#[test]
fn pack_accepts_default_single_and_multiple_precisions() {
    for (args, expected) in [
        (vec![], vec![4]),
        (vec!["--experts", "3"], vec![3]),
        (vec!["--experts", "2,3"], vec![2, 3]),
        (vec!["--experts", "4", "3", "2"], vec![4, 3, 2]),
        (vec!["--experts", "2", "--experts", "3"], vec![2, 3]),
    ] {
        let cli =
            Cli::try_parse_from(["cherenkov", "pack", "/model"].into_iter().chain(args)).unwrap();
        let Some(Command::Pack {
            experts, model_dir, ..
        }) = cli.command
        else {
            panic!("pack expected")
        };

        assert_eq!(experts, expected);
        assert_eq!(model_dir, Some(PathBuf::from("/model")));
    }
}

#[test]
fn pack_rejects_invalid_or_missing_precisions() {
    for value in ["1", "5", "2,5", "all", "-2", ""] {
        assert!(Cli::try_parse_from(["cherenkov", "pack", "--experts", value]).is_err());
    }

    assert!(Cli::try_parse_from(["cherenkov", "pack", "--experts"]).is_err());
}

#[test]
fn server_cli_records_only_explicit_overrides() {
    let cli = Cli::try_parse_from([
        "cherenkov",
        "serve",
        "/model",
        "--max-tokens",
        "64",
        "--port",
        "9090",
    ])
    .unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };
    let root = tempfile::tempdir().unwrap();
    let source = args.source(Some(root.path().to_owned())).unwrap();

    assert_eq!(source.overrides.max_tokens, Some(64));
    assert_eq!(source.overrides.experts, None);
    assert_eq!(source.overrides.no_eos, None);
    assert_eq!(source.resolve().unwrap().server.port, 9090);
}

#[test]
fn control_commands_do_not_require_a_model_or_prompt() {
    for args in [
        vec!["cherenkov", "status", "--json"],
        vec!["cherenkov", "config", "show"],
        vec![
            "cherenkov",
            "config",
            "reload",
            "--socket",
            "/private/socket",
        ],
        vec!["cherenkov", "serve", "--print-config"],
    ] {
        assert!(Cli::try_parse_from(args).is_ok());
    }
}

#[test]
fn server_overrides_preserve_explicit_zero_false_and_adaptive() {
    let cli = Cli::try_parse_from([
        "cherenkov",
        "serve",
        "--experts",
        "4",
        "--cut-weak",
        "0",
        "--drafts",
        "0",
        "--pool-gb",
        "adaptive",
        "--no-eos=false",
    ])
    .unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };
    let overrides = args
        .source(Some(tempfile::tempdir().unwrap().path().to_owned()))
        .unwrap()
        .overrides;

    assert_eq!(overrides.experts, Some(4));
    assert_eq!(overrides.cut_weak, Some(0.0));
    assert_eq!(overrides.drafts, Some(0));
    assert_eq!(
        overrides.pool_gb,
        Some(cherenkov::options::PoolBudget::Adaptive)
    );
    assert_eq!(overrides.no_eos, Some(false));

    let cli = Cli::try_parse_from(["cherenkov", "serve", "--no-eos"]).unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };

    assert_eq!(
        args.source(Some(tempfile::tempdir().unwrap().path().to_owned()))
            .unwrap()
            .overrides
            .no_eos,
        Some(true)
    );
}

#[test]
fn server_rejects_cli_only_options_and_invalid_numbers() {
    for args in [
        vec!["--raw"],
        vec!["--check"],
        vec!["--repeat", "1"],
        vec!["--experts", "1"],
        vec!["--drafts", "4"],
        vec!["--max-ctx", "0"],
        vec!["--max-tokens", "0"],
        vec!["--cut-weak", "NaN"],
        vec!["--pool-gb", "0"],
    ] {
        assert!(Cli::try_parse_from(["cherenkov", "serve"].into_iter().chain(args)).is_err());
    }
}

#[test]
fn portable_root_and_default_config_work_without_a_model_argument() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(
        dir.path().join("cherenkov.toml"),
        "[defaults]\nmax_tokens=128",
    )
    .unwrap();

    let cli = Cli::try_parse_from([
        "cherenkov",
        "serve",
        "--root",
        dir.path().to_str().unwrap(),
        "--print-config",
    ])
    .unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };
    let source = args.source(cli.root).unwrap();
    let config = source.resolve().unwrap();

    assert_eq!(config.defaults.max_tokens, 128);
    assert_eq!(config.server.root.as_deref(), Some(dir.path()));
    assert_eq!(
        config.model_dir().unwrap(),
        Paths::new(Some(dir.path())).unwrap().default_model()
    );

    for args in [
        vec!["paths"],
        vec!["download", "--metadata-only"],
        vec!["pack"],
    ] {
        assert!(Cli::try_parse_from(["cherenkov"].into_iter().chain(args)).is_ok());
    }
}
