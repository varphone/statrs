use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use distribution::Distribution;
use result::Result;
use error::StatsError;

/// Implements the [Fisher-Snedecor](https://en.wikipedia.org/wiki/F-distribution) distribution
/// also commonly known as the F-distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{FisherSnedecor, Continuous};
/// use statrs::statistics::Mean;
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FisherSnedecor {
    freedom_1: f64,
    freedom_2: f64,
}

impl FisherSnedecor {
    /// Constructs a new fisher-snedecor distribution with
    /// degrees of freedom `freedom_1` and `freedom_2`
    ///
    /// # Errors
    ///
    /// Returns an error if `freedom_1` or `freedom_2` are `NaN`.
    /// Also returns an error if `freedom_1 <= 0.0` or `freedom_2 <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let mut result = FisherSnedecor::new(1.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = FisherSnedecor::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(freedom_1: f64, freedom_2: f64) -> Result<FisherSnedecor> {
        if freedom_1.is_nan() || freedom_2.is_nan() {
            Err(StatsError::BadParams)
        } else if freedom_1 <= 0.0 || freedom_2 <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(FisherSnedecor {
                freedom_1: freedom_1,
                freedom_2: freedom_2,
            })
        }
    }

    /// Returns the first degree of freedom for the
    /// fisher-snedecor distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(2.0, 3.0).unwrap();
    /// assert_eq!(n.freedom_1(), 2.0);
    /// ```
    pub fn freedom_1(&self) -> f64 {
        self.freedom_1
    }

    /// Returns the second degree of freedom for the
    /// fisher-snedecor distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(2.0, 3.0).unwrap();
    /// assert_eq!(n.freedom_2(), 3.0);
    /// ```
    pub fn freedom_2(&self) -> f64 {
        self.freedom_2
    }
}

impl Sample<f64> for FisherSnedecor {
    /// Generate a random sample from a fisher-snedecor distribution
    /// using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details.
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for FisherSnedecor {
    /// Generate a random independent sample from a fisher-snedecor distribution
    /// using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details.
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for FisherSnedecor {
    /// Generate a random sample from a fisher-snedecor distribution using
    /// `r` as the source of randomness.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{FisherSnedecor, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = FisherSnedecor::new(2.0, 2.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        (super::gamma::sample_unchecked(r, self.freedom_1 / 2.0, 0.5) * self.freedom_2) /
        (super::gamma::sample_unchecked(r, self.freedom_2 / 2.0, 0.5) * self.freedom_1)
    }
}
 