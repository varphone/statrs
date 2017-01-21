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
    fn new(&self, p: f64) -> Result<Geometric> {
        if p <= 0.0 || p > 1.0 {
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
    fn p(&self) -> f64 {
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
        if p == 1.0 {
            1.0
        } else {
            (1.0 - r.next_f64()).log(1.0 - self.p).ceil()
        }
    }
}