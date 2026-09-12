//! Scrollable inspection uses the same field formatting as CLI reports.

use super::{
    state::{Pane, State},
    view::panel,
};
use crate::cli_output::{
    Report,
    terminal::{layout, theme::Theme},
};
use anyhow::{Context, Result};
use ratatui::{
    Frame,
    layout::{Rect, Size},
    widgets::Paragraph,
};
use tui_scrollview::{ScrollView, ScrollbarVisibility};

pub(super) struct Canvas {
    width: u16,
    view: ScrollView,
}

impl Canvas {
    fn new(report: &Report, width: u16, theme: Theme) -> Result<Self> {
        let panels = layout::panels(report, width, theme)?;
        let height: usize = panels.iter().map(|panel| usize::from(panel.height)).sum();
        let height = u16::try_from(height).context("dashboard report is too tall")?;
        let mut view = ScrollView::new(Size::new(width, height.max(1)))
            .horizontal_scrollbar_visibility(ScrollbarVisibility::Never);
        let mut y = 0;

        for panel in panels {
            let area = Rect::new(0, y, width, panel.height);
            y += panel.height;

            panel.render(area, view.buf_mut());
        }

        Ok(Self { width, view })
    }
}

pub(super) fn draw(state: &mut State, frame: &mut Frame<'_>, area: Rect) {
    let area = panel(
        frame,
        area,
        " Details / totals since load ",
        state.pane == Pane::Detail,
        state.theme,
    );

    if area.width < 2 || area.height == 0 {
        return;
    }

    let Some(snapshot) = &state.snapshot else {
        return;
    };
    let Some(selected) = state.table.selected() else {
        frame.render_widget(Paragraph::new("No rows on this page"), area);

        return;
    };
    let width = area.width - 1;

    if state
        .canvas
        .as_ref()
        .is_none_or(|canvas| canvas.width != width)
    {
        let report = snapshot.rows.detail_report(selected);

        match Canvas::new(&report, width, state.theme) {
            Ok(canvas) => state.canvas = Some(canvas),
            Err(error) => {
                frame.render_widget(Paragraph::new(super::view::clean(&error.to_string())), area);

                return;
            }
        }
    }

    if let Some(canvas) = &state.canvas {
        frame.render_stateful_widget(&canvas.view, area, &mut state.scroll);
    }
}
