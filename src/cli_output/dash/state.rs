//! Navigation and refresh state, independent of terminal I/O.

use super::{
    client::{Client, Reply, Snapshot},
    detail::Canvas,
    history::History,
};
use crate::cli_output::terminal::theme::Theme;
use anyhow::{Context, Result, bail};
use ratatui::{
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    widgets::TableState,
};
use std::{
    path::PathBuf,
    sync::mpsc::TryRecvError,
    time::{Duration, Instant},
};
use tui_scrollview::ScrollViewState;

const REFRESH: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Tab {
    #[default]
    Summary,
    Layers,
    Experts,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Pane {
    History,
    #[default]
    Rows,
    Detail,
}

impl Pane {
    fn next(self, overview: bool) -> Self {
        match self {
            Self::History => Self::Rows,
            Self::Rows => Self::Detail,
            Self::Detail if overview => Self::History,
            Self::Detail => Self::Rows,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Request {
    pub generation: u64,
    pub tab: Tab,
    pub offset: usize,
    pub layer: usize,
}

struct Location {
    request: Request,
    selected: Option<usize>,
    pages: Vec<usize>,
}

pub(super) struct State {
    pub socket: PathBuf,
    pub request: Request,
    pub snapshot: Option<Snapshot>,
    pub history: History,
    pub error: Option<String>,
    pub updated: Option<Instant>,
    pub canvas: Option<Canvas>,
    pub scroll: ScrollViewState,
    pub table: TableState,
    pub pane: Pane,
    pub expanded: bool,
    pub help: bool,
    pub theme: Theme,
    pub next_offset: Option<usize>,
    pub layer_count: Option<usize>,
    previous: Vec<usize>,
    back: Option<Location>,
    in_flight: bool,
    refresh_at: Instant,
}

impl State {
    pub fn new(socket: PathBuf, color: bool) -> Self {
        Self {
            socket,
            request: Request::default(),
            snapshot: None,
            history: History::default(),
            error: None,
            updated: None,
            canvas: None,
            scroll: ScrollViewState::default(),
            table: TableState::default(),
            pane: Pane::Rows,
            expanded: false,
            help: false,
            theme: Theme::new(color),
            next_offset: None,
            layer_count: None,
            previous: Vec::new(),
            back: None,
            in_flight: false,
            refresh_at: Instant::now(),
        }
    }

    pub fn poll(&mut self, client: &Client) -> Result<bool> {
        let changed = match client.replies.try_recv() {
            Ok(reply) => {
                self.in_flight = false;

                self.accept(reply);

                true
            }
            Err(TryRecvError::Empty) => false,
            Err(TryRecvError::Disconnected) => bail!("dashboard control worker stopped"),
        };

        if self.in_flight || Instant::now() < self.refresh_at {
            return Ok(changed);
        }

        client
            .requests
            .try_send(self.request)
            .context("dashboard control worker stopped")?;

        self.in_flight = true;

        Ok(changed)
    }

    pub fn accept(&mut self, reply: Reply) {
        if reply.request != self.request {
            return;
        }

        self.refresh_at = Instant::now() + REFRESH;

        match reply.result {
            Ok(snapshot) => {
                self.history.observe(&snapshot.summary);

                self.next_offset = snapshot.next_offset();
                self.layer_count = Some(snapshot.layer_count);

                self.table.select(
                    (snapshot.len() > 0)
                        .then(|| self.table.selected().unwrap_or(0).min(snapshot.len() - 1)),
                );

                self.snapshot = Some(snapshot);
                self.updated = Some(Instant::now());
                self.error = None;
                self.canvas = None;
            }
            Err(error) => {
                self.error = Some(format!("{error:#}"));

                self.history.disconnect();
            }
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> bool {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return true;
        }

        if key.code == KeyCode::Char('q') {
            return true;
        }

        if self.help {
            self.help = false;

            return false;
        }

        self.navigate(key.code)
    }

    fn navigate(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Esc => return self.escape(),
            KeyCode::Char('1') => self.tab(Tab::Summary),
            KeyCode::Char('2') => self.tab(Tab::Layers),
            KeyCode::Char('3') => self.tab(Tab::Experts),
            KeyCode::Tab => self.pane = self.pane.next(self.request.tab == Tab::Summary),
            KeyCode::Char('f') => self.expanded = !self.expanded,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Enter => self.inspect(),
            KeyCode::Char('n') => self.next_page(),
            KeyCode::Char('p') => self.previous_page(),
            KeyCode::Char('[') => self.layer(false),
            KeyCode::Char(']') => self.layer(true),
            KeyCode::Char('r') => self.refresh_at = Instant::now(),
            _ => self.scroll_key(key),
        }

        false
    }

    fn escape(&mut self) -> bool {
        if self.expanded {
            self.expanded = false;

            return false;
        }

        let Some(location) = self.back.take() else {
            return true;
        };
        let generation = self.request.generation;
        self.request = location.request;
        self.request.generation = generation;
        self.previous = location.pages;

        self.changed();
        self.table.select(location.selected);

        false
    }

    fn inspect(&mut self) {
        if self.request.tab == Tab::Experts {
            self.pane = Pane::Detail;
            self.expanded = true;

            return;
        }

        if self.pane != Pane::Rows {
            self.expanded = true;

            return;
        }

        let Some((layer, _)) = self
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.row(self.table.selected()?))
        else {
            return;
        };
        self.back = Some(Location {
            request: self.request,
            selected: self.table.selected(),
            pages: std::mem::take(&mut self.previous),
        });
        self.request.tab = Tab::Experts;
        self.request.layer = layer;
        self.request.offset = 0;
        self.expanded = false;

        self.changed();
    }

    fn scroll_key(&mut self, key: KeyCode) {
        if self.pane == Pane::Rows {
            self.select_row(key);

            return;
        }

        if self.pane != Pane::Detail {
            return;
        }

        match key {
            KeyCode::Up | KeyCode::Char('k') => self.scroll.scroll_up(),
            KeyCode::Down | KeyCode::Char('j') => self.scroll.scroll_down(),
            KeyCode::PageUp => self.scroll.scroll_page_up(),
            KeyCode::PageDown => self.scroll.scroll_page_down(),
            KeyCode::Home => self.scroll.scroll_to_top(),
            KeyCode::End => self.scroll.scroll_to_bottom(),
            _ => {}
        }
    }

    fn select_row(&mut self, key: KeyCode) {
        let len = self.snapshot.as_ref().map_or(0, Snapshot::len);

        if len == 0 {
            return;
        }

        let selected = self.table.selected().unwrap_or(0);
        let next = match key {
            KeyCode::Up | KeyCode::Char('k') => selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => (selected + 1).min(len - 1),
            KeyCode::Home | KeyCode::PageUp => 0,
            KeyCode::End | KeyCode::PageDown => len - 1,
            _ => return,
        };

        if selected == next {
            return;
        }

        self.table.select(Some(next));

        self.canvas = None;
        self.scroll = ScrollViewState::default();
    }

    fn tab(&mut self, tab: Tab) {
        if tab == self.request.tab {
            return;
        }

        self.request.tab = tab;
        self.request.offset = 0;
        self.pane = Pane::Rows;
        self.expanded = false;
        self.back = None;

        self.previous.clear();
        self.changed();
    }

    fn next_page(&mut self) {
        let Some(offset) = self.next_offset else {
            return;
        };

        self.previous.push(self.request.offset);

        self.request.offset = offset;

        self.changed();
    }

    fn previous_page(&mut self) {
        let Some(offset) = self.previous.pop() else {
            return;
        };
        self.request.offset = offset;

        self.changed();
    }

    fn layer(&mut self, forward: bool) {
        if self.request.tab != Tab::Experts {
            return;
        }

        let last = self.layer_count.unwrap_or(1).saturating_sub(1);
        let layer = if forward {
            self.request.layer.saturating_add(1).min(last)
        } else {
            self.request.layer.saturating_sub(1)
        };

        if layer == self.request.layer {
            return;
        }

        self.request.layer = layer;
        self.request.offset = 0;

        self.previous.clear();
        self.changed();
    }

    fn changed(&mut self) {
        self.request.generation += 1;
        self.snapshot = None;
        self.canvas = None;
        self.updated = None;
        self.error = None;
        self.next_offset = None;
        self.table = TableState::default();
        self.scroll = ScrollViewState::default();
        self.refresh_at = Instant::now();
    }
}
