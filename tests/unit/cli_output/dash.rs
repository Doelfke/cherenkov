use super::*;
use crate::cli_output::{stats::View, tests::summary};
use cherenkov::control::stats as wire;
use cherenkov::qwen4_exp::gpu::{ExpertCounters, LayerStats};
use client::{Reply, Snapshot};
use ratatui::{
    Terminal,
    backend::TestBackend,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
};
use state::{Pane, Request, Tab};

fn state() -> State {
    State::new(PathBuf::from("/tmp/control.sock"), false)
}

fn key(state: &mut State, key: char) -> bool {
    press(state, KeyCode::Char(key))
}

fn press(state: &mut State, key: KeyCode) -> bool {
    state.key(KeyEvent::new(key, KeyModifiers::NONE))
}

fn snapshot(request: Request, next_offset: Option<usize>) -> Snapshot {
    let summary: wire::Snapshot<wire::Summary> = serde_json::from_value(summary()).unwrap();
    let rows = (0..3)
        .map(|index| wire::Layer {
            layer: request.offset + index,
            name: format!("model.layers.{}", request.offset + index),
            experts: 4,
            counters: ExpertCounters {
                selected_rows: 12,
                read_bytes_requested: 1_000_000 * index as u64,
                ..Default::default()
            },
            streaming: LayerStats::default(),
        })
        .collect();

    Snapshot {
        rows: View::Layers(wire::Snapshot {
            observation: summary.observation.clone(),
            data: wire::Page {
                offset: request.offset,
                next_offset,
                total: 3,
                data: rows,
            },
        }),
        summary: Box::new(summary),
        layer_count: 3,
    }
}

fn success(request: Request, next_offset: Option<usize>) -> Reply {
    Reply {
        request,
        result: Ok(snapshot(request, next_offset)),
    }
}

fn accept(state: &mut State) {
    state.accept(success(state.request, None));
}

#[test]
fn navigation_discards_late_responses_from_other_views() {
    let mut state = state();
    let old = state.request;

    key(&mut state, '2');
    state.accept(success(old, Some(8)));

    assert!(state.snapshot.is_none());
    assert!(state.next_offset.is_none());
    assert_eq!(state.request.tab, Tab::Layers);

    accept(&mut state);
    assert!(state.snapshot.is_some());
}

#[test]
fn pages_and_layers_obey_bounds() {
    let mut state = state();

    key(&mut state, '2');
    state.accept(success(state.request, Some(8)));
    key(&mut state, 'n');
    assert_eq!(state.request.offset, 8);
    key(&mut state, 'n');
    assert_eq!(state.request.offset, 8);
    accept(&mut state);
    key(&mut state, 'p');
    assert_eq!(state.request.offset, 0);
    key(&mut state, 'p');
    assert_eq!(state.request.offset, 0);
    key(&mut state, '3');

    for _ in 0..5 {
        key(&mut state, ']');
    }

    assert_eq!(state.request.layer, 2);

    for _ in 0..5 {
        key(&mut state, '[');
    }

    assert_eq!(state.request.layer, 0);
}

#[test]
fn unavailable_server_keeps_last_snapshot_until_recovery() {
    let mut state = state();

    accept(&mut state);

    let updated = state.updated;

    state.accept(Reply {
        request: state.request,
        result: Err(anyhow::anyhow!("server unavailable")),
    });

    assert!(state.snapshot.is_some());
    assert_eq!(state.updated, updated);
    assert!(
        state
            .error
            .as_deref()
            .unwrap()
            .contains("server unavailable")
    );

    accept(&mut state);
    assert!(state.error.is_none());
}

#[test]
fn quit_and_control_c_work_without_a_server() {
    let mut state = state();

    assert!(key(&mut state, 'q'));
    assert!(state.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)));
}

fn populated_screen() -> (State, Terminal<TestBackend>) {
    let mut state = state();

    accept(&mut state);

    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();

    terminal.draw(|frame| state.draw(frame)).unwrap();

    (state, terminal)
}

fn screen(terminal: &Terminal<TestBackend>) -> String {
    terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

#[test]
fn dashboard_has_charts_pinned_headers_and_details() {
    let (state, terminal) = populated_screen();
    let text = screen(&terminal);

    assert!(text.contains("Expert reads"));
    assert!(text.contains("CPU read wait"));
    assert!(text.contains("GPU stage coverage"));
    assert!(text.contains("Read MB"));
    assert!(text.contains("Details / totals"));
    assert_eq!(state.table.selected(), Some(0));
}

#[test]
fn tiny_terminals_and_every_focus_expand_without_panicking() {
    let (mut state, mut terminal) = populated_screen();

    for (width, height) in [(80, 24), (40, 12), (8, 5), (2, 2)] {
        terminal.backend_mut().resize(width, height);

        for pane in [Pane::History, Pane::Rows, Pane::Detail] {
            state.pane = pane;

            terminal.draw(|frame| state.draw(frame)).unwrap();
            key(&mut state, 'f');
            terminal.draw(|frame| state.draw(frame)).unwrap();
            assert!(!press(&mut state, KeyCode::Esc));
        }
    }
}

#[test]
fn refresh_preserves_row_selection_and_detail_scroll() {
    let (mut state, mut terminal) = populated_screen();

    key(&mut state, 'j');
    press(&mut state, KeyCode::Tab);
    key(&mut state, 'j');
    terminal.draw(|frame| state.draw(frame)).unwrap();

    let scroll = state.scroll.offset();

    accept(&mut state);
    terminal.draw(|frame| state.draw(frame)).unwrap();

    assert_eq!(state.table.selected(), Some(1));
    assert_eq!(state.scroll.offset(), scroll);
}

#[test]
fn inspect_and_back_restore_layer_selection_and_page() {
    let mut state = state();

    accept(&mut state);
    key(&mut state, 'j');
    press(&mut state, KeyCode::Enter);

    assert_eq!(state.request.tab, Tab::Experts);
    assert_eq!(state.request.layer, 1);
    assert!(!press(&mut state, KeyCode::Esc));

    accept(&mut state);
    assert_eq!(state.request.tab, Tab::Summary);
    assert_eq!(state.table.selected(), Some(1));
}

#[test]
fn help_and_expansion_do_not_change_requests() {
    let mut state = state();
    let request = state.request;

    key(&mut state, '?');
    assert!(state.help);
    press(&mut state, KeyCode::Esc);
    assert!(!state.help);
    key(&mut state, 'f');
    assert!(state.expanded);
    press(&mut state, KeyCode::Esc);
    assert!(!state.expanded);
    assert_eq!(state.request, request);
}

#[test]
fn polling_a_slow_socket_does_not_block_navigation() {
    use std::io::{BufRead, BufReader, Write};

    use std::os::unix::net::UnixListener;

    use std::sync::mpsc;

    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let (started, ready) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let mock = snapshot(Request::default(), None);
    let View::Layers(layers) = mock.rows else {
        unreachable!()
    };
    let responses = [
        serde_json::to_value(mock.summary).unwrap(),
        serde_json::to_value(layers).unwrap(),
    ];
    let server = std::thread::spawn(move || {
        for (index, response) in responses.into_iter().enumerate() {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();

            BufReader::new(&stream).read_line(&mut request).unwrap();

            if index == 0 {
                started.send(request).unwrap();
                wait.recv().unwrap();
            }

            writeln!(
                stream,
                "{}",
                serde_json::json!({"ok": true, "data": response})
            )
            .unwrap();
        }
    });
    let client = Client::start(socket).unwrap();
    let mut state = state();

    state.poll(&client).unwrap();
    assert!(
        ready
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .contains("stats_summary")
    );
    state.poll(&client).unwrap();
    key(&mut state, '2');
    assert!(key(&mut state, 'q'));
    release.send(()).unwrap();
    server.join().unwrap();

    let reply = client.replies.recv_timeout(Duration::from_secs(2)).unwrap();

    assert!(reply.result.is_ok());
    state.accept(reply);
    assert!(state.snapshot.is_none());
}

#[test]
fn expert_inspection_preserves_full_counter_values() {
    let mut state = state();

    key(&mut state, '3');

    let mut snapshot = snapshot(state.request, None);
    snapshot.rows = View::Experts(wire::Snapshot {
        observation: snapshot.summary.observation.clone(),
        data: wire::Page {
            offset: 0,
            next_offset: None,
            total: 1,
            data: vec![wire::Expert {
                layer: 0,
                expert: 7,
                counters: ExpertCounters {
                    selected_rows: u64::MAX,
                    ..Default::default()
                },
            }],
        },
    });

    state.accept(Reply {
        request: state.request,
        result: Ok(snapshot),
    });
    press(&mut state, KeyCode::Enter);

    assert_eq!(state.pane, Pane::Detail);
    assert!(state.expanded);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

    terminal.draw(|frame| state.draw(frame)).unwrap();

    assert!(screen(&terminal).contains(&u64::MAX.to_string()));
}

#[test]
fn empty_page_clears_selection_and_focus_is_visible_without_color() {
    let (mut state, mut terminal) = populated_screen();
    let mut snapshot = snapshot(state.request, None);
    let View::Layers(page) = &mut snapshot.rows else {
        unreachable!()
    };

    page.data.data.clear();

    page.data.total = 0;

    state.accept(Reply {
        request: state.request,
        result: Ok(snapshot),
    });
    press(&mut state, KeyCode::Tab);
    terminal.draw(|frame| state.draw(frame)).unwrap();

    assert!(state.table.selected().is_none());
    assert!(screen(&terminal).contains("> Details"));
    assert!(screen(&terminal).contains("No rows on this page"));
}
