//! Bounded control requests run off the terminal event thread.

use super::state::{Request, Tab};
use crate::cli_output::stats::View;
use anyhow::{Result, ensure};
use cherenkov::control::{self, Command, stats as wire};
use cherenkov::qwen4_exp::gpu::ExpertCounters;
use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
};

const PAGE_SIZE: usize = 8;

pub(super) struct Client {
    pub requests: SyncSender<Request>,
    pub replies: Receiver<Reply>,
}

pub(super) struct Reply {
    pub request: Request,
    pub result: Result<Snapshot>,
}

pub(super) struct Snapshot {
    pub summary: Box<wire::Snapshot<wire::Summary>>,
    pub rows: View,
    pub layer_count: usize,
}

impl Snapshot {
    pub fn next_offset(&self) -> Option<usize> {
        match &self.rows {
            View::Layers(page) => page.data.next_offset,
            View::Experts(page) => page.data.next_offset,
            View::Summary(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        match &self.rows {
            View::Layers(page) => page.data.data.len(),
            View::Experts(page) => page.data.data.len(),
            View::Summary(_) => 0,
        }
    }

    pub fn row(&self, index: usize) -> Option<(usize, &ExpertCounters)> {
        match &self.rows {
            View::Layers(page) => page
                .data
                .data
                .get(index)
                .map(|row| (row.layer, &row.counters)),
            View::Experts(page) => page
                .data
                .data
                .get(index)
                .map(|row| (row.expert, &row.counters)),
            View::Summary(_) => None,
        }
    }
}

impl Client {
    pub fn start(socket: PathBuf) -> Result<Self> {
        let (requests, incoming) = mpsc::sync_channel::<Request>(1);
        let (outgoing, replies) = mpsc::sync_channel(1);

        thread::Builder::new()
            .name("dashboard-control".into())
            .spawn(move || {
                while let Ok(request) = incoming.recv() {
                    let result = fetch(&socket, request);

                    if outgoing.send(Reply { request, result }).is_err() {
                        break;
                    }
                }
            })?;

        // Closing the channels releases the worker after any pending query's
        // protocol deadline. Terminal exit never waits for socket I/O.
        Ok(Self { requests, replies })
    }
}

fn query(socket: &Path, command: Command) -> Result<View> {
    View::from_response(&command, &control::query(socket, command.clone())?)
}

fn fetch(socket: &Path, request: Request) -> Result<Snapshot> {
    let View::Summary(summary) = query(socket, Command::StatsSummary)? else {
        unreachable!("summary query returns a summary view")
    };
    let experts = request.tab == Tab::Experts;
    let View::Layers(layers) = query(
        socket,
        Command::StatsLayers {
            offset: if experts { 0 } else { request.offset },
            limit: if experts { 1 } else { PAGE_SIZE },
        },
    )?
    else {
        unreachable!("layer query returns a layer view")
    };
    let layer_count = layers.data.total;
    let rows = if experts {
        ensure!(
            request.layer < layer_count,
            "layer {} is unavailable; return to Layers",
            request.layer
        );

        query(
            socket,
            Command::StatsExperts {
                layer: request.layer,
                offset: request.offset,
                limit: PAGE_SIZE,
            },
        )?
    } else {
        View::Layers(layers)
    };

    Ok(Snapshot {
        summary,
        rows,
        layer_count,
    })
}
