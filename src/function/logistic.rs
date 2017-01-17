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

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    #[test]
    fn test_logistic() {
        for x in 0i64..10 {
            let xf64 = x as f64;
            assert!(super::logistic(xf64) == 1.0 / ((-xf64).exp() + 1.0));
        }
    }

    #[test]
    fn test_logit() {
        let mut x = 0.1;
        for _ in 0..9 {
            assert!(super::logit(x) == (x / (1.0 - x)).ln());
            x += 0.1;
        }
    }
}