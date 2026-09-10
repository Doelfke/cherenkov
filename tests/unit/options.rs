use super::*;
use clap::Parser;

#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    options: Options,
}

#[test]
fn flags_resolve_defaults_and_mixed_precision() {
    let a = TestCli::try_parse_from(["test"]).unwrap().options;

    assert_eq!((a.experts, a.miss_bits(), a.drafts), (4, 4, 2));
    a.validate().unwrap();

    for bits in ["2", "3"] {
        let a = TestCli::try_parse_from(["test", "--experts", bits])
            .unwrap()
            .options;

        assert_eq!(a.experts, a.miss_bits());
        a.validate().unwrap();

        let a = TestCli::try_parse_from(["test", "--miss-experts", bits])
            .unwrap()
            .options;

        assert_eq!(a.experts, 4);
        a.validate().unwrap();
    }

    let a = TestCli::try_parse_from(["test", "--experts", "3", "--miss-experts", "2"])
        .unwrap()
        .options;

    assert!(a.validate().is_err());
}

#[test]
fn invalid_numbers_fail_at_the_interface() {
    for (flag, value) in [
        ("--experts", "1"),
        ("--drafts", "4"),
        ("--cut-weak", "NaN"),
        ("--cut-weak", "1.1"),
        ("--pool-gb", "0"),
        ("--pool-gb", "inf"),
        ("--repeat", "0"),
        ("--max-ctx", "0"),
        ("--max-tokens", "0"),
    ] {
        assert!(
            TestCli::try_parse_from(["test", flag, value]).is_err(),
            "{flag} {value}"
        );
    }

    assert_eq!("max".parse::<PoolBudget>().unwrap(), PoolBudget::Max);
}
