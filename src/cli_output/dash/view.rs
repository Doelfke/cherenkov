//! A fixed dashboard shell with focusable history, rows and inspection panes.

use super::{
    detail, plots, rows,
    state::{Pane, State, Tab},
};
use crate::cli_output::terminal::theme::Theme;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Margin, Rect},
    text::Line,
    widgets::{Block, Clear, Paragraph, Tabs, Wrap},
};

impl State {
    pub fn draw(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area().inner(Margin {
            horizontal: 1,
            vertical: 0,
        });
        let [tabs, status, body, help] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .areas(area);
        let selected = match self.request.tab {
            Tab::Summary => 0,
            Tab::Layers => 1,
            Tab::Experts => 2,
        };
        let tabs_widget = Tabs::new(["1 Overview", "2 Layers", "3 Experts"])
            .select(selected)
            .style(self.theme.label)
            .highlight_style(self.theme.heading)
            .divider("  ");

        frame.render_widget(tabs_widget, tabs);
        frame.render_widget(
            Paragraph::new(self.status()).style(self.theme.label),
            status,
        );
        self.draw_body(frame, body);

        let keys = if help.width < 60 {
            "Tab focus  ? help  q quit"
        } else {
            "Tab focus  j/k move  Enter inspect  f expand  ? help  q quit"
        };

        frame.render_widget(Paragraph::new(keys).style(self.theme.label), help);

        if self.help {
            self.draw_help(frame, area);
        }
    }

    fn draw_body(&mut self, frame: &mut Frame<'_>, area: Rect) {
        if area.width < 2 || area.height == 0 {
            return;
        }

        // Small terminals show only the focused pane. Tab still reaches each
        // pane without squeezing headers or tables into unusable rectangles.
        if self.expanded || area.width < 65 || area.height < 12 {
            self.draw_pane(frame, area, self.pane);

            return;
        }

        let body = if self.request.tab == Tab::Summary {
            let [charts, body] =
                Layout::vertical([Constraint::Length(8), Constraint::Min(0)]).areas(area);

            plots::draw(self, frame, charts);

            body
        } else {
            area
        };
        let [list, inspect] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Fill(1)]).areas(body);

        rows::draw(self, frame, list);
        detail::draw(self, frame, inspect);
    }

    fn draw_pane(&mut self, frame: &mut Frame<'_>, area: Rect, pane: Pane) {
        match pane {
            Pane::History => plots::draw(self, frame, area),
            Pane::Rows => rows::draw(self, frame, area),
            Pane::Detail => detail::draw(self, frame, area),
        }
    }

    fn status(&self) -> String {
        if let Some(error) = &self.error {
            return format!("Unavailable: {}", clean(error));
        }

        let Some(updated) = self.updated else {
            return "Connecting...".into();
        };
        let position = if self.request.tab == Tab::Experts {
            format!("layer {} | ", self.request.layer)
        } else {
            String::new()
        };

        format!(
            "{position}updated {}s ago | {}",
            updated.elapsed().as_secs(),
            clean(&self.socket.display().to_string())
        )
    }

    fn draw_help(&self, frame: &mut Frame<'_>, area: Rect) {
        let width = area.width.min(66);
        let height = area.height.min(22);
        let area = Rect::new(
            area.x + (area.width - width) / 2,
            area.y + (area.height - height) / 2,
            width,
            height,
        );

        frame.render_widget(Clear, area);

        let inner = panel(frame, area, " Help / any key closes ", true, self.theme);
        let lines = [
            "1/2/3       Overview / Layers / Experts",
            "Tab         Focus the next pane",
            "j/k, arrows Select a row or scroll details",
            "Home/End    First/last row or top/bottom of details",
            "PgUp/PgDn   First/last row or scroll details by page",
            "Enter       Inspect the selected layer's experts",
            "f           Expand / restore the focused pane",
            "Escape      Restore pane, return from inspection, or quit",
            "n/p         Next / previous server page",
            "[ / ]       Previous / next expert layer",
            "r           Refresh now",
            "q, Ctrl-C   Quit",
            "",
            "Charts use interval changes over up to two minutes.",
            "GPU coverage is stage time / measured GPU phase time.",
            "CPU read wait combines demand and prefetch waits.",
            "Tables show totals. Read bars scale to the current page.",
            "Gaps indicate missing measurements. Queries are read-only.",
        ];

        frame.render_widget(
            Paragraph::new(lines.map(Line::raw).to_vec()).wrap(Wrap { trim: false }),
            inner,
        );
    }
}

pub(super) fn panel(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    focused: bool,
    theme: Theme,
) -> Rect {
    let border = if focused { theme.heading } else { theme.rule };
    let title = if focused {
        format!(">{title}")
    } else {
        title.to_owned()
    };
    let block = Block::bordered().title(title).border_style(border);
    let inner = block.inner(area);

    frame.render_widget(block, area);

    inner
}

pub(super) fn clean(text: &str) -> String {
    text.chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect()
}
