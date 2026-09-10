//! Keep long transfers visible without flooding stderr with per-chunk updates.

use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

#[derive(Default)]
pub(super) struct Reporter {
    last: Mutex<Option<Instant>>,
}

impl ProgressHandler for Reporter {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                let mut last = self.last.lock().unwrap();
                if last.is_none_or(|t| t.elapsed() >= Duration::from_secs(2)) {
                    eprintln!(
                        "download {:.2}/{:.2} GB",
                        *bytes_completed as f64 / 1e9,
                        *total_bytes as f64 / 1e9
                    );
                    *last = Some(Instant::now());
                }
            }
            DownloadEvent::Complete => eprintln!("download complete"),
            _ => {}
        }
    }
}
