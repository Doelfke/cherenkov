//! Print reusable Ratatui report panels into normal terminal scrollback.

pub(super) mod layout;
pub(super) mod theme;

use super::Report;
use anyhow::{Context, Result};
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    backend::{Backend, CrosstermBackend},
};
use std::io;
use theme::Theme;

pub(super) fn print(report: &Report) -> Result<()> {
    let stdout = io::stdout();
    let color = anstream::AutoStream::choice(&stdout) != anstream::ColorChoice::Never;
    let backend = CrosstermBackend::new(stdout.lock());
    // Reserve a blank line for the shell prompt. Raw mode and the alternate
    // screen are unnecessary for a report that is printed once.
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(1),
        },
    )?;
    let width = terminal.size()?.width.max(1);

    append(&mut terminal, report, width, color)
}

pub(super) fn append<B: Backend>(
    terminal: &mut Terminal<B>,
    report: &Report,
    width: u16,
    color: bool,
) -> Result<()>
where
    B::Error: std::error::Error + Send + Sync + 'static,
{
    let margin = match width {
        0..12 => 0,
        12..60 => 1,
        _ => 2,
    };
    let content_width = width.saturating_sub(margin * 2).clamp(1, 100);
    let panels = layout::panels(report, content_width, Theme::new(color))?;

    let height: usize = panels.iter().map(|panel| usize::from(panel.height)).sum();
    let height = u16::try_from(height).context("report is too tall for terminal rendering")?;

    terminal.insert_before(height, |buffer| {
        let mut y = buffer.area.y;

        for panel in panels {
            let mut area = buffer.area;

            area.x += margin;
            area.y = y;
            area.width = content_width;
            area.height = panel.height;
            y += panel.height;

            panel.render(area, buffer);
        }
    })?;

    let prompt = terminal.get_frame().area().as_position();

    terminal.set_cursor_position(prompt)?;
    terminal.backend_mut().flush()?;

    Ok(())
}
