//! HTTP status for expected lifecycle failures; validation defaults to 400.

#[derive(Debug)]
pub(super) struct Failure(pub u16, pub &'static str);

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.1)
    }
}

impl std::error::Error for Failure {}

pub(super) fn status(error: &anyhow::Error) -> u16 {
    error.downcast_ref::<Failure>().map_or(400, |e| e.0)
}
