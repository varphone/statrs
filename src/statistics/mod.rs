//! Provides traits for statistical computation

pub use self::statistics::*;
pub use self::inplace_statistics::*;

mod statistics;
mod inplace_statistics;
mod slice_statistics;
