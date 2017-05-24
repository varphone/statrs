use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use statistics::*;
use {Result, StatsError};

/// Implements the [Erlang](https://en.wikipedia.org/wiki/Erlang_distribution) distribution
/// which is a special case of the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Erlang, Continuous};
/// use statrs::statistics::Mean;
/// use statrs::prec;
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Erlang {
    shape: u64,
    rate: f64,
}

impl Erlang {
    /// Constructs a new erlang distribution with a shape (k)
    /// of `shape` and a rate (λ) of `rate`
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` or `rate` are `NaN`.
    /// Also returns an error if `shape == 0` or `rate <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Erlang;
    ///
    /// let mut result = Erlang::new(3, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Erlang::new(0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(shape: u64, rate: f64) -> Result<Erlang> {
        if rate.is_nan() || shape == 0 || rate <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Erlang { shape: shape, rate: rate })
        }
    }

    /// Returns the shape (k) of the erlang distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Erlang;
    ///
    /// let n = Erlang::new(3, 1.0).unwrap();
    /// assert_eq!(n.shape(), 3);
    /// ```
    pub fn shape(&self) -> u64 {
        self.shape
    }

    /// Returns the rate (λ) of the erlang distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Erlang;
    ///
    /// let n = Erlang::new(3, 1.0).unwrap();
    /// assert_eq!(n.rate(), 1.0);
    /// ```
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Sample<f64> for Erlang {
    /// Generate a random sample from a erlang
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Erlang {
    /// Generate a random independent sample from a erlang
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for Erlang {
    /// Generate a random sample from a erlang distribution using
    /// `r` as the source of randomness.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Erlang, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Erlang::new(3, 1.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::gammma::sample_unchecked(r, self.shape, self.rate)
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use distribution::{Univariate, Continuous, Erlang};

    fn try_create(shape: u64, rate: f64) -> Erlang {
        let n = Erlang::new(shape, rate);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(shape: u64, rate: f64) {
        let n = try_create(shape, rate);
        assert_eq!(shape, n.shape());
        assert_eq!(rate, n.rate());
    }

    fn bad_create_case(shape: u64, rate: f64) {
        let n = Erlang::new(shape, rate);
        assert!(n.is_err());
    }

    #[test]
    fn test_create() {
        create_case(1, 0.1);
        create_case(1, 1.0);
        create_case(10, 10.0);
        create_case(10, 1.0);
        create_case(10, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0, 1.0);
        bad_create_case(1, 0.0);
        bad_create_case(1, f64::NAN);
        bad_create_case(1, -1.0);
    }
}