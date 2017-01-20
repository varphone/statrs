use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use function::gamma;
use statistics::*;
use distribution::{Univariate, Continuous, Distribution};
use result::Result;
use error::StatsError;

/// Implements the [Inverse Gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) distribution
///
/// # Examples
///
/// ```
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct InverseGamma {
    shape: f64,
    rate: f64,
}

impl InverseGamma {
    /// Constructs a new inverse gamma distribution with a shape (α)
    /// of `shape` and a rate (β) of `rate`
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` or `rate` are `NaN`.
    /// Also returns an error if `shape <= 0.0` or `rate <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGamma;
    ///
    /// let mut result = InverseGamma::new(3.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = InverseGamma::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(shape: f64, rate: f64) -> Result<InverseGamma> {
        let is_nan = shape.is_nan() || rate.is_nan();
        match (shape, rate, is_nan) {
            (_, _, true) => Err(StatsError::BadParams),
            (_, _, false) if shape <= 0.0 || rate <= 0.0 => Err(StatsError::BadParams),
            (_, _, false) => {
                Ok(InverseGamma {
                    shape: shape,
                    rate: rate,
                })
            }
        }
    }

    /// Returns the shape (α) of the inverse gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGamma;
    ///
    /// let n = InverseGamma::new(3.0, 1.0).unwrap();
    /// assert_eq!(n.shape(), 3.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns the rate (β) of the inverse gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGamma;
    ///
    /// let n = InverseGamma::new(3.0, 1.0).unwrap();
    /// assert_eq!(n.rate(), 1.0);
    /// ```
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Sample<f64> for InverseGamma {
    /// Generate a random sample from an inverse gamma
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for InverseGamma {
    /// Generate a random independent sample from an inverse gamma
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for InverseGamma {
    /// Generate a random sample from an inverse gamma distribution
    /// using `r` as the source of randomness.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{InverseGamma, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = InverseGamma::new(3.0, 1.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        1.0 / super::gamma::sample_unchecked(r, self.shape, self.rate)
    }
}

impl Univariate<f64, f64> for InverseGamma {
    /// Calculates the cumulative distribution function for the inverse gamma
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Γ(α, β / x) / Γ(α)
    /// ```
    ///
    /// where the numerator is the upper incomplete gamma function,
    /// the denominator is the gamma function, `α` is the shape,
    /// and `β` is the rate
    fn cdf(&self, x: f64) -> f64 {
        gamma::gamma_ur(self.shape, self.rate / x)
    }
}

impl Min<f64> for InverseGamma {
    /// Returns the minimum value in the domain of the
    /// inverse gamma distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> f64 {
        0.0
    }
}

impl Max<f64> for InverseGamma {
    /// Returns the maximum value in the domain of the
    /// inverse gamma distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// INF
    /// ```
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use statistics::*;
    use distribution::{Univariate, Continuous, InverseGamma};

    fn try_create(shape: f64, rate: f64) -> InverseGamma {
        let n = InverseGamma::new(shape, rate);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(shape: f64, rate: f64) {
        let n = try_create(shape, rate);
        assert_eq!(shape, n.shape());
        assert_eq!(rate, n.rate());
    }

    fn bad_create_case(shape: f64, rate: f64) {
        let n = InverseGamma::new(shape, rate);
        assert!(n.is_err());
    }

    fn get_value<F>(shape: f64, rate: f64, eval: F) -> f64
        where F: Fn(InverseGamma) -> f64
    {
        let n = try_create(shape, rate);
        eval(n)
    }

    fn test_case<F>(shape: f64, rate: f64, expected: f64, eval: F)
        where F: Fn(InverseGamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(shape: f64, rate: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(InverseGamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(shape: f64, rate: f64, eval: F)
        where F: Fn(InverseGamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert!(x.is_nan());
    }

    #[test]
    fn test_create() {
        create_case(0.1, 0.1);
        create_case(1.0, 1.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-100.0, 1.0);
        bad_create_case(f64::NEG_INFINITY, 1.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, 0.0);
        bad_create_case(1.0, -1.0);
        bad_create_case(1.0, -100.0);
        bad_create_case(1.0, f64::NEG_INFINITY);
        bad_create_case(1.0, f64::NAN);
    }

    #[test]
    fn test_cdf() {
        test_almost(0.1, 0.1, 0.1862151961946054271994, 1e-14, |x| x.cdf(1.2));
        test_almost(0.1, 1.0, 0.05859755410986647796141, 1e-14, |x| x.cdf(2.0));
        test_case(0.1, f64::INFINITY, 0.0, |x| x.cdf(1.1));
        test_case(1.0, 0.1, 0.9355069850316177377304, |x| x.cdf(1.5));
        test_almost(1.0, 1.0, 0.4345982085070782231613, 1e-14, |x| x.cdf(1.2));
        test_case(1.0, f64::INFINITY, 0.0, |x| x.cdf(1.5));
        test_is_nan(f64::INFINITY, 0.1, |x| x.cdf(5.0));
        test_is_nan(f64::INFINITY, 1.0, |x| x.cdf(2.5));
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.cdf(1.0));
    }

    #[test]
    fn test_min_max() {
        test_case(1.0, 1.0, 0.0, |x| x.min());
        test_case(1.0, 1.0, f64::INFINITY, |x| x.max());
    }
}