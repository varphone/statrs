use std::i64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use statistics::*;
use distribution::{Univariate, Discrete, Distribution};
use result::Result;
use error::StatsError;

/// Implements the [Geometric](https://en.wikipedia.org/wiki/Geometric_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Geometric {
    p: f64,
}

impl Geometric {
    /// Constructs a new geometric distribution with a probability
    /// of `p`
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is not in `(0, 1]`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Geometric;
    ///
    /// let mut result = Geometric::new(0.5);
    /// assert!(result.is_ok());
    ///
    /// result = Geometric::new(0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: f64) -> Result<Geometric> {
        if p <= 0.0 || p > 1.0 || p.is_nan() {
            Err(StatsError::BadParams)
        } else {
            Ok(Geometric { p: p })
        }
    }

    /// Returns the probability `p` of the geometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Geometric;
    ///
    /// let n = Geometric::new(0.5).unwrap();
    /// assert_eq!(n.p(), 0.5);
    /// ```
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Sample<f64> for Geometric {
    /// Generate a random sample from a geometric
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Geometric {
    /// Generate a random independent sample from a geometric
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for Geometric {
    /// Generates a random sample from the geometric distribution
    /// using `r` as the source of randomness
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Geometric, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Geometric::new(0.5).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        if self.p == 1.0 {
            1.0
        } else {
            (1.0 - r.next_f64()).log(1.0 - self.p).ceil()
        }
    }
}

impl Univariate<i64, f64> for Geometric {
    fn cdf(&self, x: f64) -> f64 {
        1.0 - (1.0 - self.p).powf(x)
    }
}

impl Min<i64> for Geometric {
    /// Returns the minimum value in the domain of the
    /// geometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1
    /// ```
    fn min(&self) -> i64 {
        1
    }
}

impl Max<i64> for Geometric {
    /// Returns the maximum value in the domain of the
    /// geometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2^63 - 1
    /// ```
    fn max(&self) -> i64 {
        i64::MAX
    }
}

impl Mean<f64> for Geometric {
    /// Returns the mean of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / p
    /// ```
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
}

impl Variance<f64> for Geometric {
    /// Returns the standard deviation of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 - p) / p^2
    /// ```
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }

    /// Returns the standard deviation of the geometric distribution
    ///
    /// # Remarks
    ///
    /// Returns `NAN` if `p` is `1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(1 - p) / p
    /// ```
    fn std_dev(&self) -> f64 {
        (1.0 - self.p).sqrt() / self.p
    }
}

impl Entropy<f64> for Geometric {
    /// Returns the entropy of the geometric distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (-(1 - p) * log_2(1 - p) - p * log_2(p)) / p
    /// ```
    fn entropy(&self) -> f64 {
        (-self.p * self.p.log(2.0) - (1.0 - self.p) * (1.0 - self.p).log(2.0)) / self.p
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::{i64, f64};
    use statistics::*;
    use distribution::{Univariate, Discrete, Geometric};

    fn try_create(p: f64) -> Geometric {
        let n = Geometric::new(p);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(p: f64) {
        let n = try_create(p);
        assert_eq!(p, n.p());
    }

    fn bad_create_case(p: f64) {
        let n = Geometric::new(p);
        assert!(n.is_err());
    }

    fn get_value<T, F>(p: f64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Geometric) -> T
    {
        let n = try_create(p);
        eval(n)
    }

    fn test_case<T, F>(p: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Geometric) -> T
    {
        let x = get_value(p, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(p: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Geometric) -> f64
    {
        let x = get_value(p, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(p: f64, eval: F)
        where F: Fn(Geometric) -> f64
    {
        let x = get_value(p, eval);
        assert!(x.is_nan());
    }

    #[test]
    fn test_create() {
        create_case(0.3);
        create_case(1.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
        bad_create_case(0.0);
        bad_create_case(-1.0);
        bad_create_case(2.0);
    }

    #[test]
    fn test_mean() {
        test_case(0.3, 1.0 / 0.3, |x| x.mean());
        test_case(1.0, 1.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(0.3, 0.7 / (0.3 * 0.3), |x| x.variance());
        test_case(1.0, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(0.3, 0.7f64.sqrt() / 0.3, |x| x.std_dev());
        test_case(1.0, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(0.3, 2.937636330768973333333, 1e-14, |x| x.entropy());
        test_is_nan(1.0, |x| x.entropy());
    }

    #[test]
    fn test_min_max() {
        test_case(0.3, 1, |x| x.min());
        test_case(0.3, i64::MAX, |x| x.max());
    }

    #[test]
    fn test_cdf() {
        test_case(0.3, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 1.0, |x| x.cdf(1.0));
        test_case(1.0, 1.0, |x| x.cdf(2.0));
    }
}