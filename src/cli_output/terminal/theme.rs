//! Semantic styles shared by report panels.

use ratatui::style::{Color, Style};

#[derive(Clone, Copy)]
pub(in crate::cli_output) struct Theme {
    pub heading: Style,
    pub label: Style,
    pub header: Style,
    pub rule: Style,
    pub selected: Style,
    pub series: [Style; 3],
}

impl Theme {
    pub fn new(color: bool) -> Self {
        if !color {
            return Self {
                heading: Style::default(),
                label: Style::default(),
                header: Style::default(),
                rule: Style::default(),
                selected: Style::default(),
                series: [Style::default(); 3],
            };
        }

        // Body text and rules inherit the terminal foreground so light and
        // dark palettes remain readable. Only section headings use an accent.
        Self {
            heading: Style::new().fg(Color::Cyan).bold(),
            label: Style::new().dim(),
            header: Style::new().bold(),
            rule: Style::new().dim(),
            selected: Style::new().reversed(),
            series: [Color::Cyan, Color::Yellow, Color::Magenta]
                .map(|color| Style::new().fg(color)),
        }
    }
}
