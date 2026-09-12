use anyhow::Result;
use std::{fs, process::Command};
use xtask::authors;

#[test]
fn author_entries_match_complete_logins_and_require_name_and_email() -> Result<()> {
    let file = tempfile::NamedTempFile::new()?;
    let cases = [
        ("Example Person <contributor@example.org> (@contributor)", true),
        (
            "Example Person <123+contributor@users.noreply.github.com> (@contributor)",
            true,
        ),
        (
            "Example Person <contributor@users.noreply.github.com> (@contributor)",
            true,
        ),
        ("  Example Person <dev@example.org> (@CONTRIBUTOR)\r\n", true),
        (
            "Terms of contribution.\n\nExample Person <dev@example.org> (@contributor)\n",
            true,
        ),
        ("Other Contributor <dev@example.org> (@other-contributor)", false),
        ("Example Person <dev@example.org> (@contributor-extra)", false),
        ("Example Person (@contributor)", false),
        ("<dev@example.org> (@contributor)", false),
        ("Example Person <not-an-email> (@contributor)", false),
        ("Please add yourself (@contributor)", false),
        ("", false),
    ];

    for (text, expected) in cases {
        fs::write(file.path(), text)?;

        assert_eq!(
            authors::check(file.path(), "contributor").is_ok(),
            expected,
            "{text:?}"
        );
    }

    Ok(())
}

#[test]
fn missing_entry_explains_how_to_acknowledge_the_terms() -> Result<()> {
    let file = tempfile::NamedTempFile::new()?;
    let error = authors::check(file.path(), "new-contributor")
        .unwrap_err()
        .to_string();

    assert!(error.contains("Read the contribution terms in AUTHORS"));
    assert!(error.contains("Your Name <your Git email> (@new-contributor)"));
    assert!(error.contains("noreply"));

    Ok(())
}

#[test]
fn missing_authors_file_reports_the_path() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("AUTHORS");
    let error = authors::check(&path, "contributor").unwrap_err().to_string();

    assert!(error.contains(path.to_str().unwrap()));

    Ok(())
}

#[test]
fn cli_returns_failure_for_an_unlisted_login() -> Result<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["check-author", "unlisted-contributor-for-test"])
        .output()?;

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("(@unlisted-contributor-for-test)"));

    Ok(())
}
