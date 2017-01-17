//! Provides the [logistic](http://en.wikipedia.org/wiki/Logistic_function) and related functions

use error::StatsError;

/// Computes the logistic function
pub fn logistic(p: f64) -> f64 {
    1.0 / ((-p).exp() + 1.0)
}

/// Computes the logit function
///
/// # Panics
///
/// If `p <= 0.0` or `p >= 1.0`
pub fn logit(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0,
            format!("{}", StatsError::ArgIntervalExcl("p", 0.0, 1.0)));
    (p / (1.0 - p)).ln()
}