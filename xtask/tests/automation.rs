use anyhow::Result;
use serde_json::json;
use std::fs;
use xtask::{bench, metrics, prefill, report, suite, util};

const TELEMETRY: &str = include_str!("fixtures/telemetry.txt");

#[test]
fn loading_is_separate_from_inference() -> Result<()> {
    let m = metrics::parse(TELEMETRY)?;

    assert_eq!(
        (
            m["load_seconds"].as_f64(),
            m["prefill_seconds"].as_f64(),
            m["decode_seconds"].as_f64()
        ),
        (Some(1.18), Some(5.0), Some(8.0))
    );
    assert!(m["store_build_seconds"].is_null());
    assert_eq!(
        metrics::parse(&format!("built the 2-bit store in 75s\n{TELEMETRY}"))?["store_build_seconds"],
        75.0
    );
    assert_eq!(
        (
            m["pp_s"].as_f64(),
            m["tg_s"].as_f64(),
            m["metal_gb"].as_f64()
        ),
        (Some(4.8), Some(8.0), Some(20.98))
    );
    assert!(
        metrics::parse(&TELEMETRY.replace("prefill 24", "cached 24"))
            .unwrap_err()
            .to_string()
            .contains("prefill")
    );

    Ok(())
}

#[test]
fn svg_preserves_bytes_and_rejects_invalid_xml() -> Result<()> {
    let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M 0 0 L 10 10\"/></svg>";
    let text = format!("reasoning\n```svg\n{svg}\n```");

    assert_eq!(metrics::extract_svg(&text)?, (svg, 2));
    assert!(metrics::extract_svg(&svg[..svg.len() - 6]).is_err());
    assert!(metrics::extract_svg("<svg><path></svg>").is_err());

    Ok(())
}

#[test]
fn schedule_rotates_configs_and_keeps_art_last() -> Result<()> {
    let s: suite::Suite =
        serde_json::from_value(util::json(&util::root().join("benchmarks/suite.json"))?)?;
    let jobs = suite::schedule(&s.configurations, &s.cases, s.rounds);

    assert_eq!(jobs.len(), 80);
    assert_eq!(s.configurations[jobs[0].1].id, "exact-4bit");
    assert_eq!(
        s.configurations[jobs.iter().find(|j| j.0 == 1).unwrap().1].id,
        "misses-2bit"
    );
    assert!(
        jobs[jobs.len() - 4..]
            .iter()
            .all(|j| s.cases[j.2].kind == "svg")
    );
    assert_eq!(
        jobs.iter()
            .filter(|j| s.cases[j.2].id == "prefill-long")
            .count(),
        4
    );

    Ok(())
}

#[test]
fn cycle_check_preserves_advancing_numbers() {
    let block = (0..40).map(|i| format!("word{i} ")).collect::<String>();

    assert!(metrics::repeated_tail(&(block.repeat(5) + "partial")).is_some());
    assert!(metrics::repeated_tail(&block.repeat(2)).is_none());

    let advancing = (1..101)
        .map(|i| format!("cache.put({i}, {i}) # Evicts {}\n", i - 3))
        .collect::<String>();

    assert!(metrics::repeated_tail(&advancing).is_none());
}

#[test]
fn resume_only_allows_larger_caps() {
    let old = json!({"binary_sha256":"same","suite_sha256":"same","cases":[{"id":"code","max_tokens":4096,"prompt":"original"}]});
    let mut new = old.clone();
    new["cases"][0]["max_tokens"] = json!(7168);

    assert!(suite::only_higher_caps(&old, &new));
    assert!(!suite::only_higher_caps(&new, &old));

    new["cases"][0]["prompt"] = json!("different");

    assert!(!suite::only_higher_caps(&old, &new));

    new = old.clone();
    new["binary_sha256"] = json!("different");

    assert!(!suite::only_higher_caps(&old, &new));
}

#[test]
fn larger_caps_archive_only_incomplete_outputs() -> Result<()> {
    let out = tempfile::tempdir()?;
    let before = json!({"cases":[{"id":"code","max_tokens":4096}]});
    let after = json!({"cases":[{"id":"code","max_tokens":7168}]});
    let completed = json!({"id":"done","case":"code","status":"ok","args":["--max-tokens","4096"],"output":"outputs/done.txt"});
    let capped = json!({"id":"capped","case":"code","status":"incomplete","args":["--max-tokens","4096"],"output":"outputs/capped.txt"});

    fs::create_dir(out.path().join("outputs"))?;
    fs::write(out.path().join("outputs/done.txt"), b"complete answer\n")?;
    fs::write(
        out.path().join("outputs/capped.txt"),
        b"answer cut mid-sentence",
    )?;

    let mut report = json!({"signature":before,"cases":before["cases"],"runs":[completed,capped]});

    bench::raise_saved_caps(out.path(), &mut report, after.clone())?;
    assert_eq!(report["runs"], json!([completed]));
    assert_eq!(
        fs::read(out.path().join("outputs/done.txt"))?,
        b"complete answer\n"
    );

    let archived = &report["previous_attempts"][0];

    assert_eq!(
        fs::read(out.path().join(archived["output"].as_str().unwrap()))?,
        b"answer cut mid-sentence"
    );
    assert_eq!(archived["args"], json!(["--max-tokens", "4096"]));
    assert_eq!(report["suite_revisions"][0]["previous_signature"], before);
    assert_eq!(report["signature"], after);

    Ok(())
}

#[test]
fn invalid_samples_do_not_enter_reports() -> Result<()> {
    let out = tempfile::tempdir()?;
    let data = json!({"provenance":{"commit":"test"},"cases":[{"id":"code"}],"configurations":[{"id":"exact","label":"4-bit"}],"runs":[{"id":"ok","case":"code","configuration":"exact","status":"ok","metrics":metrics::parse(TELEMETRY)?},{"id":"bad","case":"code","configuration":"exact","status":"failed","error":"power source changed"}]});

    report::write(out.path(), &data)?;

    let summary = fs::read_to_string(out.path().join("summary.md"))?;

    assert!(summary.contains("| code | 4-bit | 1 | 64 | 8.00 | 1.18 | 4.8 | 8.00 | 20.98 |"));
    assert!(summary.contains("bad: power source changed"));

    let html = fs::read_to_string(out.path().join("gallery.html"))?;

    assert!(html.contains("8.00</td>"));
    assert!(!html.contains("{{"));

    let saved = format!("{}\n", serde_json::to_string(&data)?);

    fs::write(out.path().join("report.json"), &saved)?;
    report::regenerate(out.path(), &data)?;
    assert_eq!(fs::read_to_string(out.path().join("report.json"))?, saved);

    Ok(())
}

#[test]
fn mixed_prefill_does_not_change_q4_controls() -> Result<()> {
    let out = tempfile::tempdir()?;
    let mut runs = Vec::new();

    for (miss, rates, texts) in [
        (4, [20, 20], ["same", "same"]),
        (2, [100, 150], ["original", "changed"]),
    ] {
        for (i, label) in ["before", "after"].iter().enumerate() {
            let file = format!("q4-miss{miss}-{label}.txt");

            fs::write(out.path().join(&file), texts[i])?;

            let mut r = json!({"bits":4,"binary":label,"prompt":"short","round":1,"valid":true,"output":file,"metrics":{"prompt_tokens":121,"pp_s":rates[i]}});

            if miss != 4 {
                r["miss_bits"] = json!(miss);
            }

            runs.push(r);
        }
    }

    let mut invalid = runs.last().unwrap().clone();
    invalid["valid"] = json!(false);
    invalid["metrics"]["pp_s"] = json!(999);

    runs.push(invalid);

    let observations = "# Run observations\n\nThese observations belong to the author.\n";

    fs::write(out.path().join("README.md"), observations)?;
    prefill::write_summary(out.path(), &json!({"runs":runs,"prompts":{"short":"test"}}))?;

    let summary = fs::read_to_string(out.path().join("summary.md"))?;

    assert_eq!(
        fs::read_to_string(out.path().join("README.md"))?,
        observations
    );
    assert!(summary.contains("[Run observations](README.md)"));

    assert!(summary.contains("| 121 | 4 | 20.0 | 20.0 | +0.0% | 1 |"));
    assert!(summary.contains("| 121 | 4 / 2 | 100.0 | 150.0 | +50.0% | 1 |"));
    assert_eq!(summary.matches("the Q4 outputs are identical").count(), 1);
    assert!(!summary.contains("the Q4 outputs differ"));

    Ok(())
}

#[test]
fn all_saved_reports_render_and_keep_sample_counts() -> Result<()> {
    let root = util::root();
    let report = util::json(&root.join("results/baseline-2026-09-09/report.json"))?;
    let rows = report::rows(&report)?;

    assert_eq!(rows.len(), 32);
    assert_eq!(
        rows.iter()
            .map(|r| r["runs"].as_u64().unwrap())
            .sum::<u64>(),
        80
    );

    let out = tempfile::tempdir()?;

    report::write(out.path(), &report)?;

    let html = fs::read_to_string(out.path().join("gallery.html"))?;

    assert_eq!(html.matches("<img ").count(), 4);
    assert!(!html.contains("<svg"));

    let summary = fs::read_to_string(out.path().join("summary.md"))?;

    assert_eq!(summary.matches("![Pelican]").count(), 4);
    assert!(summary.contains("## Pelicans\n"));
    assert!(summary.contains("## Answers\n"));
    assert!(!summary.contains("gallery.html"));
    assert!(!summary.contains("https://"));

    for run in report["runs"].as_array().unwrap() {
        let output = run["output"].as_str().unwrap();

        assert!(
            summary.contains(&format!("]({output})")),
            "missing {output}"
        );
    }

    Ok(())
}

#[test]
fn sha256_matches_standard_vector() -> Result<()> {
    let file = tempfile::NamedTempFile::new()?;

    fs::write(file.path(), b"abc")?;
    assert_eq!(
        util::digest(file.path())?,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    Ok(())
}

#[test]
fn selection_rejects_duplicates_and_unknown_ids() {
    let items = vec!["a".to_owned(), "b".to_owned()];

    assert!(suite::select(&items, Some("a,a"), |s| s.as_str()).is_err());
    assert!(suite::select(&items, Some("c"), |s| s.as_str()).is_err());
    assert_eq!(
        suite::select(&items, Some("b,a"), |s| s.as_str()).unwrap(),
        vec!["b", "a"]
    );
}
