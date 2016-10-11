//! Provides traits for statistical computation

pub use self::traits::*;
pub use self::statistics::*;
pub use self::inplace_statistics::*;
pub use self::iter_statistics::*;

mod traits;
mod statistics;
mod inplace_statistics;
mod iter_statistics;
mod slice_statistics;
