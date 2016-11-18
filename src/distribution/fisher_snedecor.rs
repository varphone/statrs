use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use function::beta;
use statistics::*;
use distribution::{Univariate, Continuous, Distribution};
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
/// use statrs::prec;
///
/// let n = FisherSnedecor::new(3.0, 3.0).unwrap();
/// assert_eq!(n.mean(), 3.0);
/// assert!(prec::almost_eq(n.pdf(1.0), 0.318309886183790671538, 1e-15));
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

impl Univariate<f64, f64> for FisherSnedecor {
    /// Calculates the cumulative distribution function for the fisher-snedecor distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// I_((d1 * x) / (d1 * x + d2))(d1 / 2, d2 / 2)
    /// ```
    ///
    /// where `d1` is the first degree of freedom, `d2` is
    /// the second degree of freedom, and `I` is the regularized incomplete
    /// beta function
    fn cdf(&self, x: f64) -> f64 {
        beta::beta_reg(self.freedom_1 / 2.0,
                       self.freedom_2 / 2.0,
                       self.freedom_1 * x / (self.freedom_1 * x + self.freedom_2))
    }
}

impl Min<f64> for FisherSnedecor {
    /// Returns the minimum value in the domain of the
    /// fisher-snedecor distribution representable by a double precision
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

impl Max<f64> for FisherSnedecor {
    /// Returns the maximum value in the domain of the
    /// fisher-snedecor distribution representable by a double precision
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

impl Mean<f64> for FisherSnedecor {
    /// Returns the mean of the fisher-snedecor distribution
    ///
    /// # Panics
    ///
    /// If `freedom_2 <= 2.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// d2 / (d2 - 2)
    /// ```
    ///
    /// where `d2` is the second degree of freedom
    fn mean(&self) -> f64 {
        assert!(self.freedom_2 > 2.0, StatsError::ArgGt("freedom_2", 2.0));
        self.freedom_2 / (self.freedom_2 - 2.0)
    }
}

impl Variance<f64> for FisherSnedecor {
    /// Returns the variance of the fisher-snedecor distribution
    ///
    /// # Panics
    ///
    /// If `freedom_2 <= 4.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2 * d2^2 * (d1 + d2 - 2)) / (d1 * (d2 - 2)^2 * (d2 - 4))
    /// ```
    ///
    /// where `d1` is the first degree of freedom and `d2` is
    /// the second degree of freedom
    fn variance(&self) -> f64 {
        assert!(self.freedom_2 > 4.0, StatsError::ArgGt("freedom_2", 4.0));
        (2.0 * self.freedom_2 * self.freedom_2 * (self.freedom_1 + self.freedom_2 - 2.0)) /
        (self.freedom_1 * (self.freedom_2 - 2.0) * (self.freedom_2 - 2.0) * (self.freedom_2 - 4.0))
    }

    /// Returns the standard deviation of the fisher-snedecor distribution
    ///
    /// # Panics
    ///
    /// If `freedom_2 <= 4.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt((2 * d2^2 * (d1 + d2 - 2)) / (d1 * (d2 - 2)^2 * (d2 - 4)))
    /// ```
    ///
    /// where `d1` is the first degree of freedom and `d2` is
    /// the second degree of freedom
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Skewness<f64> for FisherSnedecor {
    /// Returns the skewness of the fisher-snedecor distribution
    ///
    /// # Panics
    ///
    /// If `freedom_2 <= 6.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ((2d1 + d2 - 2) * sqrt(8 * (d2 - 4))) / ((d2 - 6) * sqrt(d1 * (d1 + d2 - 2)))
    /// ```
    ///
    /// where `d1` is the first degree of freedom and `d2` is
    /// the second degree of freedom
    fn skewness(&self) -> f64 {
        assert!(self.freedom_2 > 6.0, StatsError::ArgGt("freedom_2", 6.0));
        ((2.0 * self.freedom_1 + self.freedom_2 - 2.0) * (8.0 * (self.freedom_2 - 4.0)).sqrt()) /
        ((self.freedom_2 - 6.0) * (self.freedom_1 * (self.freedom_1 + self.freedom_2 - 2.0)).sqrt())
    }
}

impl Mode<f64> for FisherSnedecor {
    /// Returns the mode for the fisher-snedecor distribution
    ///
    /// # Panics
    ///
    /// If `freedom_1 <= 2.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ((d1 - 2) / d1) * (d2 / (d2 + 2))
    /// ```
    ///
    /// where `d1` is the first degree of freedom and `d2` is
    /// the second degree of freedom
    fn mode(&self) -> f64 {
        assert!(self.freedom_1 > 2.0, StatsError::ArgGt("freedom_1", 2.0));
        (self.freedom_2 * (self.freedom_1 - 2.0)) / (self.freedom_1 * (self.freedom_2 + 2.0))
    }
}

impl Continuous<f64, f64> for FisherSnedecor {
    /// Calculates the probability density function for the fisher-snedecor distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(((d1 * x) ^ d1 * d2 ^ d2) / (d1 * x + d2) ^ (d1 + d2)) / (x * β(d1 / 2, d2 / 2))
    /// ```
    ///
    /// where `d1` is the first degree of freedom, `d2` is
    /// the second degree of freedom, and `β` is the beta function
    fn pdf(&self, x: f64) -> f64 {
        ((self.freedom_1 * x).powf(self.freedom_1) * self.freedom_2.powf(self.freedom_2) /
         (self.freedom_1 * x + self.freedom_2).powf(self.freedom_1 + self.freedom_2))
            .sqrt() / (x * beta::beta(self.freedom_1 / 2.0, self.freedom_2 / 2.0))
    }

    /// Calculates the log probability density function for the fisher-snedecor distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(sqrt(((d1 * x) ^ d1 * d2 ^ d2) / (d1 * x + d2) ^ (d1 + d2)) / (x * β(d1 / 2, d2 / 2)))
    /// ```
    ///
    /// where `d1` is the first degree of freedom, `d2` is
    /// the second degree of freedom, and `β` is the beta function
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}