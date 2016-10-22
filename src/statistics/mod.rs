//! Provides traits for statistical computation

pub use self::traits::*;
pub use self::statistics::*;
pub use self::order_statistics::*;
pub use self::iter_statistics::*;

mod traits;
mod statistics;
mod order_statistics;
mod iter_statistics;
mod slice_statistics;
