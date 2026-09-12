//! Native chart widgets over a two-minute window of interval samples.

use super::{
    history::History,
    state::{Pane, State},
    view::panel,
};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    symbols::Marker,
    widgets::{Axis, Chart, Dataset, GraphType, Paragraph},
};

const LABELS: [(&str, &str); 3] = [
    ("Expert reads", "MB/s"),
    ("CPU read wait", "ms/s"),
    ("GPU stage coverage", "%"),
];

pub(super) fn draw(state: &State, frame: &mut Frame<'_>, area: Rect) {
    let areas = if area.width >= 65 {
        Layout::horizontal([Constraint::Fill(1); 3]).areas::<3>(area)
    } else {
        Layout::vertical([Constraint::Fill(1); 3]).areas::<3>(area)
    };

    for (index, area) in areas.into_iter().enumerate() {
        draw_chart(state, frame, area, index);
    }
}

fn draw_chart(state: &State, frame: &mut Frame<'_>, area: Rect, index: usize) {
    let (label, unit) = LABELS[index];
    let latest = state
        .history
        .samples
        .back()
        .and_then(|sample| sample.values[index]);
    let value = latest.map_or_else(|| "n/a".into(), |value| format!("{value:.1}"));
    let title = format!(" {label} ");
    let inner = panel(
        frame,
        area,
        &title,
        state.pane == Pane::History,
        state.theme,
    );
    let [number, plot] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(inner);
    let stale = if state.error.is_some() {
        " (stale)"
    } else {
        ""
    };

    frame.render_widget(
        Paragraph::new(format!("{value} {unit}{stale}")).style(state.theme.series[index]),
        number,
    );

    if plot.height < 3 || plot.width < 8 {
        return;
    }

    let segments = segments(&state.history, index);
    let max = if index == 2 {
        100.0
    } else {
        segments
            .iter()
            .flatten()
            .map(|point| point.1)
            .fold(1.0, f64::max)
            * 1.1
    };
    let datasets: Vec<_> = segments
        .iter()
        .map(|data| {
            Dataset::default()
                .data(data)
                .graph_type(GraphType::Line)
                .marker(Marker::Braille)
                .style(state.theme.series[index])
        })
        .collect();
    let chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .bounds([-120.0, 0.0])
                .labels(["-120s", "now"])
                .style(state.theme.rule),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, max])
                .labels(["0".to_owned(), format!("{max:.0}")])
                .style(state.theme.rule),
        );

    frame.render_widget(chart, plot);
}

fn segments(history: &History, index: usize) -> Vec<Vec<(f64, f64)>> {
    let end = history.samples.back().map_or(0.0, |sample| sample.time);
    let mut segments = Vec::new();
    let mut segment = Vec::new();

    for sample in &history.samples {
        if let Some(value) = sample.values[index] {
            segment.push((sample.time - end, value));
        } else if !segment.is_empty() {
            segments.push(std::mem::take(&mut segment));
        }
    }

    if !segment.is_empty() {
        segments.push(segment);
    }

    segments
}
