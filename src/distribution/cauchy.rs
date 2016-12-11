use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use statistics::*;
use distribution::{Univariate, Continuous, Distribution};
use result::Result;
use error::StatsError;

/// Implements the [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
/// distribution, also known as the Lorentz distribution.
///
/// # Examples
///
/// ```
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Cauchy {
    location: f64,
    scale: f64,
}

impl Cauchy {
    /// Constructs a new cauchy distribution with the given
    /// location and scale.
    ///
    /// # Errors
    ///
    /// Returns an error if location is `NaN` or `rate <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let mut result = Cauchy::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Cauchy::new(0.0, -1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64) -> Result<Cauchy> {
        if location.is_nan() || scale <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Cauchy {
                location: location,
                scale: scale,
            })
        }
    }

    /// Returns the location of the cauchy distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let n = Cauchy::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the cauchy distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let n = Cauchy::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Sample<f64> for Cauchy {
    /// Generate a random sample from a cauchy
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Cauchy {
    /// Generate a random independent sample from a cauchy
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for Cauchy {
    /// Generate a random sample from the cauchy distribution
    /// using `r` as the source of randomness
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Cauchy, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Cauchy.new(0.0, 1.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        self.location + self.scale * (f64::consts::PI * (r.next_f64() - 0.5)).tan()
    }
}

impl Univariate<f64, f64> for Cauchy {
    /// Calculates the cumulative distribution function for the
    /// cauchy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / π) * arctan((x - x_0) / γ) + 0.5
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn cdf(&self, x: f64) -> f64 {
        (1.0 / f64::consts::PI) * ((x - self.location) / self.scale).atan() + 0.5
    }
}

impl Min<f64> for Cauchy {
    /// Returns the minimum value in the domain of the cauchy
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// NEG_INF
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for Cauchy {
    /// Returns the maximum value in the domain of the cauchy
    /// distribution representable by a double precision float
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

impl Entropy<f64> for Cauchy {
    /// Returns the entropy of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(γ) + ln(4π)
    /// ```
    ///
    /// where `γ` is the scale
    fn entropy(&self) -> f64 {
        (4.0 * f64::consts::PI * self.scale).ln()
    }
}

impl Median<f64> for Cauchy {
    /// Returns the median of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_0
    /// ```
    ///
    /// where `x_0` is the location
    fn median(&self) -> f64 {
        self.location
    }
}

impl Mode<f64> for Cauchy {
    /// Returns the mode of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_0
    /// ```
    ///
    /// where `x_0` is the location
    fn mode(&self) -> f64 {
        self.location
    }
}

impl Continuous<f64, f64> for Exponential {
    /// Calculates the probability density function for the cauchy
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / (πγ * (1 + ((x - x_0) / γ)^2))
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn pdf(&self, x: f64) -> f64 {
        1.0 /
        (f64::consts::PI * self.scale *
         (1.0 + ((x - self.location) / self.scale) * ((x - self.location) / self.scale)))
    }

    /// Calculates the log probability density function for the cauchy
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(1 / (πγ * (1 + ((x - x_0) / γ)^2)))
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        -(f64::consts::PI * self.scale *
          (1.0 + ((x - self.location) / self.scale) * ((x - self.location) / self.scale)))
            .ln()
    }
}