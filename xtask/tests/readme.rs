use anyhow::Result;
use serde_json::json;
use std::path::Path;
use xtask::{readme, util};

fn baseline() -> Result<serde_json::Value> {
    util::json(&util::root().join("results/baseline-2026-09-09/report.json"))
}

#[test]
fn saved_rates_and_all_pelicans_are_rendered() -> Result<()> {
    let data = baseline()?;
    let text = readme::render(&data, Path::new("results/baseline-2026-09-09"))?;
    assert!(text.contains("Saved revision `93c514f`; 80 valid samples."));
    assert!(text.contains("| 4-bit | 8.52 | 7.07 | 7.06 | 7.71 | 6.77 | 7.88 |"));
    assert!(text.contains("| 3-bit | 12.13 | 10.72 | 10.27 | 9.83 | 10.86 | 12.67 |"));
    assert!(text.contains("prefill-long pp/s"));
    assert_eq!(text.matches("![Pelican]").count(), 4);
    assert!(text.contains("results/baseline-2026-09-09/pelicans/exact-4bit.svg"));
    assert!(text.contains("This baseline used Q4 batched prefill"));
    assert!(!text.contains("TG/s") && !text.contains("PP/s"));
    Ok(())
}

#[test]
fn changing_data_changes_readme_and_invalid_runs_stay_excluded() -> Result<()> {
    let mut data = baseline()?;
    for run in data["runs"].as_array_mut().unwrap() {
        if run["case"] == "code" && run["configuration"] == "exact-4bit" {
            run["metrics"]["tg_s"] = json!(42.0);
        }
        if run["svg"].is_string() && run["configuration"] == "all-2bit" {
            run["status"] = json!("invalid_svg");
        }
    }
    let mut failed = data["runs"][0].clone();
    failed["status"] = json!("failed");
    failed["metrics"]["tg_s"] = json!(999.0);
    data["runs"].as_array_mut().unwrap().push(failed);
    let text = readme::render(&data, Path::new("results/test"))?;
    assert!(text.contains("| 4-bit | 42.00 |"));
    assert!(!text.contains("999.00"));
    assert_eq!(text.matches("![Pelican]").count(), 3);
    Ok(())
}

#[test]
fn section_replacement_is_repeatable_and_preserves_handwritten_text() -> Result<()> {
    let original = "# Intro\n\n<!-- benchmarks:start -->\nold\n<!-- benchmarks:end -->\n\n## Engine\nHandwritten.\n";
    let updated = readme::replace_section(original, "| Experts | code tg/s |\n")?;
    assert!(updated.starts_with("# Intro\n\n<!-- benchmarks:start -->"));
    assert!(updated.ends_with("<!-- benchmarks:end -->\n\n## Engine\nHandwritten.\n"));
    assert_eq!(
        readme::replace_section(&updated, "| Experts | code tg/s |\n")?,
        updated
    );
    assert!(readme::replace_section("no markers", "test").is_err());
    assert!(
        readme::replace_section("<!-- benchmarks:end --><!-- benchmarks:start -->", "test")
            .is_err()
    );
    Ok(())
}

#[test]
fn incomplete_reports_cannot_replace_the_readme() -> Result<()> {
    let mut data = baseline()?;
    data.as_object_mut().unwrap().remove("utc_finished");
    assert!(readme::render(&data, Path::new("results/test")).is_err());
    Ok(())
}
