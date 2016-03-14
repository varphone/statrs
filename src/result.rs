use std::result;
use error::StatsError;

/// Result type for the statrs library package that returns
/// either a result type T or a StatsError
pub type Result<T> = result::Result<T, StatsError>;
