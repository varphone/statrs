use std::error::Error;
use std::fmt;

/// Enumeration of possible errors thrown
/// within the statrs library
pub enum StatsError {
    BadParams,
    ArgMustBePositive(&'static str),
    ArgNotNegative(&'static str),
    ArgIntervalIncl(&'static str, f64, f64),
    ArgIntervalExcl(&'static str, f64, f64),
    ArgIntervalExclMin(&'static str, f64, f64),
    ArgIntervalExclMax(&'static str, f64, f64),
}

impl Error for StatsError {
    fn description(&self) -> &str {
        "Error performing statistical calculation"
    }
}

impl fmt::Display for StatsError{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            StatsError::BadParams => write!(f, "Bad distribution parameters"),
            StatsError::ArgMustBePositive(s) => write!(f, "Argument {} must be positive", s),
            StatsError::ArgNotNegative(s) => write!(f, "Argument {} must be non-negative", s),
            StatsError::ArgIntervalIncl(s, min, max) => write!(f, "Argument {} not within interval [{}, {}]", s, min, max),
            StatsError::ArgIntervalExcl(s, min, max) => write!(f, "Argument {} not within interval ({}, {})", s, min, max),
            StatsError::ArgIntervalExclMin(s, min, max) => write!(f, "Argument {} not within interval ({}, {}]", s, min, max),
            StatsError::ArgIntervalExclMax(s, min, max) => write!(f, "Argument {} not within interval [{}, {})", s, min, max),
        }
    }
}

impl fmt::Debug for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            StatsError::BadParams => write!(f, "Bad distribution parameters"),
            StatsError::ArgMustBePositive(s) => write!(f, "Argument {} must be positive", s),
            StatsError::ArgNotNegative(s) => write!(f, "Argument {} must be non-negative", s),
            StatsError::ArgIntervalIncl(s, min, max) => write!(f, "Argument {} not within interval [{}, {}]", s, min, max),
            StatsError::ArgIntervalExcl(s, min, max) => write!(f, "Argument {} not within interval ({}, {})", s, min, max),
            StatsError::ArgIntervalExclMin(s, min, max) => write!(f, "Argument {} not within interval ({}, {}]", s, min, max),
            StatsError::ArgIntervalExclMax(s, min, max) => write!(f, "Argument {} not within interval [{}, {})", s, min, max),
        }
    }
}