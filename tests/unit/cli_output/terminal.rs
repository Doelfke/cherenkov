use super::*;
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    backend::TestBackend,
    buffer::Buffer,
    style::{Color, Modifier},
};

fn lines(buffer: &Buffer) -> Vec<String> {
    buffer
        .content
        .chunks(usize::from(buffer.area.width))
        .map(|row| {
            row.iter()
                .map(|cell| cell.symbol())
                .collect::<String>()
                .trim_end()
                .to_owned()
        })
        .collect()
}

fn screen(report: &Report, width: u16, color: bool) -> TestBackend {
    let backend = TestBackend::new(width, 12);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(1),
        },
    )
    .unwrap();

    terminal::append(&mut terminal, report, width, color).unwrap();

    let prompt = terminal.get_frame().area().as_position();

    assert_eq!(terminal.get_cursor_position().unwrap(), prompt);
    assert_eq!(terminal.backend().buffer()[prompt].symbol(), " ");

    terminal.backend().clone()
}

fn history(backend: &TestBackend) -> Vec<String> {
    let mut result = lines(backend.scrollback());

    result.extend(lines(backend.buffer()));

    result
}

fn sample_table() -> Report {
    Report {
        sections: vec![Section::Table {
            title: "Reads".into(),
            data: TableData {
                headers: vec!["Source".into(), "Count".into()],
                rows: vec![
                    vec!["Demand".into(), "500".into()],
                    vec!["Prefetch".into(), "2".into()],
                ],
            },
        }],
    }
}

#[test]
fn table_has_quiet_rules_and_right_aligned_numbers() {
    let backend = screen(&sample_table(), 80, true);
    let cells = &backend.buffer().content;
    let lines = history(&backend);
    let output = lines.join("\n");

    assert!(
        cells
            .iter()
            .any(|cell| cell.symbol() == "R" && cell.fg == Color::Cyan)
    );
    assert!(
        cells
            .iter()
            .any(|cell| cell.symbol() == "─" && cell.modifier.contains(Modifier::DIM))
    );
    assert!(output.contains("  Reads"));
    assert!(!output.contains('┌'));
    assert!(!output.contains("Metric"));

    let demand = lines.iter().find(|line| line.contains("Demand")).unwrap();
    let prefetch = lines.iter().find(|line| line.contains("Prefetch")).unwrap();

    assert!(demand.ends_with("500"));
    assert!(prefetch.ends_with('2'));
    assert_eq!(demand.len(), prefetch.len());
}

#[test]
fn no_color_removes_all_terminal_styling() {
    let backend = screen(&sample_table(), 80, false);

    assert!(backend.buffer().content.iter().all(|cell| {
        cell.fg == Color::Reset && cell.bg == Color::Reset && cell.modifier.is_empty()
    }));
}

#[test]
fn reports_longer_than_the_screen_retain_every_line_in_order() {
    let expected: Vec<String> = (0..500).map(|i| format!("line {i}")).collect();
    let mut report = Report::default();

    for line in &expected {
        report.note(line.clone());
    }

    let backend = screen(&report, 40, false);
    let output: Vec<String> = history(&backend)
        .into_iter()
        .map(|line| line.trim().to_owned())
        .filter(|line| !line.is_empty())
        .collect();

    assert_eq!(output, expected);
}

#[test]
fn narrow_reports_keep_field_names_and_large_values() {
    let mut expert = serde_json::to_value(ExpertCounters {
        selected_rows: u64::MAX,
        ..Default::default()
    })
    .unwrap();
    expert["layer"] = json!(0);
    expert["expert"] = json!(1);
    let command = Command::StatsExperts {
        layer: 0,
        offset: 0,
        limit: 1,
    };
    let report = stats::View::from_response(&command, &page(json!([expert]), None))
        .unwrap()
        .report();

    for width in [40, 80] {
        let output = history(&screen(&report, width, false)).join("\n");

        assert!(output.contains(&u64::MAX.to_string()), "{output}");
        assert!(output.contains("Requested bytes"));
        assert!(output.contains("Expert"));
    }
}

#[test]
fn summary_is_a_compact_field_list_without_repeated_headers() {
    let report = stats::View::from_response(&Command::StatsSummary, &summary())
        .unwrap()
        .report();
    let lines = history(&screen(&report, 80, false));
    let output = lines.join("\n");

    assert!(output.contains("Observation"));
    assert!(output.contains("Expert precision"));
    assert!(output.contains("500.000"));
    assert!(!output.contains("Metric"));
    assert!(!output.contains('│'));
    assert!(lines.len() < 60, "{} lines: {output}", lines.len());
}

#[test]
fn wrapped_status_text_keeps_every_word() {
    let mut report = Report::default();

    report.sections.push(Section::Fields {
        title: "Observation".into(),
        values: vec![(
            "GPU timing".into(),
            "failed: timestamp allocation was unavailable".into(),
        )],
    });

    let output = history(&screen(&report, 32, false)).join(" ");

    for word in ["failed:", "timestamp", "allocation", "was", "unavailable"] {
        assert!(output.contains(word), "{output}");
    }
}
