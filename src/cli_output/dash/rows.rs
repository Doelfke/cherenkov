//! Selectable counters with read-volume bars scaled to the current page.

use super::{
    history::BYTES_PER_MB,
    state::{Pane, State, Tab},
    view::panel,
};
use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    text::Line,
    widgets::{Cell, Gauge, Row, Table},
};

pub(super) fn draw(state: &mut State, frame: &mut Frame<'_>, area: Rect) {
    let kind = if state.request.tab == Tab::Experts {
        "Experts"
    } else {
        "Layers"
    };
    let title = format!(" {kind} / totals / offset {} ", state.request.offset);
    let area = panel(frame, area, &title, state.pane == Pane::Rows, state.theme);
    let Some(snapshot) = &state.snapshot else {
        return;
    };
    let bars = area.width >= 46;
    let rows: Vec<Row> = (0..snapshot.len())
        .filter_map(|index| {
            let (id, counters) = snapshot.row(index)?;
            let lookups = counters.cache_hits as f64 + counters.cache_misses as f64;
            let hits = if lookups > 0.0 {
                format!("{:.1}%", 100.0 * counters.cache_hits as f64 / lookups)
            } else {
                "n/a".into()
            };
            let bytes = if bars {
                String::new()
            } else {
                format!("{:.1}", counters.read_bytes_requested as f64 / BYTES_PER_MB)
            };

            Some(Row::new(
                [
                    id.to_string(),
                    counters.selected_rows.to_string(),
                    hits,
                    bytes,
                ]
                .map(|text| Cell::from(Line::raw(text).right_aligned())),
            ))
        })
        .collect();
    let widths = [
        Constraint::Length(5),
        Constraint::Min(8),
        Constraint::Length(8),
        Constraint::Length(12),
    ];
    let table = Table::new(rows, widths)
        .header(
            Row::new(
                ["ID", "Rows", "Hit %", "Read MB"]
                    .map(|text| Cell::from(Line::raw(text).right_aligned())),
            )
            .style(state.theme.header),
        )
        .row_highlight_style(state.theme.selected)
        .highlight_symbol("> ")
        .column_spacing(1);

    frame.render_stateful_widget(table, area, &mut state.table);

    if bars {
        draw_bars(state, frame, area);
    }
}

fn draw_bars(state: &State, frame: &mut Frame<'_>, area: Rect) {
    let Some(snapshot) = &state.snapshot else {
        return;
    };
    let max = (0..snapshot.len())
        .filter_map(|index| snapshot.row(index))
        .map(|(_, counters)| counters.read_bytes_requested)
        .max()
        .unwrap_or(0)
        .max(1);
    let offset = state.table.offset();

    for row in 0..usize::from(area.height.saturating_sub(1)) {
        let Some((_, counters)) = snapshot.row(offset + row) else {
            break;
        };
        let value = counters.read_bytes_requested;
        let gauge = Gauge::default()
            .ratio(value as f64 / max as f64)
            .label(format!("{:.1}", value as f64 / BYTES_PER_MB))
            .gauge_style(state.theme.series[0]);
        let rect = Rect::new(area.right() - 12, area.y + 1 + row as u16, 12, 1);

        frame.render_widget(gauge, rect);
    }
}
