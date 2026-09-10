//! Byte units used by memory limits, storage sizes and progress reporting.

pub(crate) const BYTES_PER_KIB: usize = 1024;
pub(crate) const BYTES_PER_MIB: usize = 1024 * BYTES_PER_KIB;

// User-facing GB limits and transfer rates use decimal units.
pub(crate) const BYTES_PER_GB: usize = 1_000_000_000;
