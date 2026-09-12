//! Read-only dashboard over the local control protocol.

mod client;
mod detail;
mod history;
mod plots;
mod rows;
mod state;
mod view;

use anyhow::{Result, ensure};
use client::Client;
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use state::State;
use std::{
    io::{self, IsTerminal},
    path::PathBuf,
    time::{Duration, Instant},
};

pub(crate) fn run(socket: PathBuf) -> Result<()> {
    ensure!(
        io::stdout().is_terminal() && io::stdin().is_terminal(),
        "dash requires a terminal; use stats summary or stats summary --json for redirected output"
    );

    let color = anstream::AutoStream::choice(&io::stdout()) != anstream::ColorChoice::Never;
    let client = Client::start(socket.clone())?;
    let mut state = State::new(socket, color);

    // Ratatui restores raw mode, the alternate screen and the cursor on exit
    // and installs its panic restoration hook.
    ratatui::run(|terminal| -> Result<()> {
        let mut redraw = true;
        let mut last_draw = Instant::now();

        loop {
            redraw |= state.poll(&client)?;

            if redraw || last_draw.elapsed() >= Duration::from_secs(1) {
                terminal.draw(|frame| state.draw(frame))?;

                last_draw = Instant::now();
                redraw = false;
            }

            if !event::poll(Duration::from_millis(100))? {
                continue;
            }

            let event = event::read()?;
            redraw = true;

            if let Event::Key(key) = event
                && key.kind != KeyEventKind::Release
                && state.key(key)
            {
                return Ok(());
            }
        }
    })
}

#[cfg(test)]
#[path = "../../tests/unit/cli_output/dash.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/cli_output/history.rs"]
mod history_tests;
