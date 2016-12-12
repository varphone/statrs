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
/// use statrs::distribution::{Cauchy, Continuous};
/// use statrs::statistics::Mode;
///
/// let n = Cauchy::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.1591549430918953357689);
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
    /// Returns an error if location or scale are `NaN` or `scale <= 0.0`
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
        if location.is_nan() || scale.is_nan() || scale <= 0.0 {
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
    /// let n = Cauchy::new(0.0, 1.0).unwrap();
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

impl Continuous<f64, f64> for Cauchy {
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

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use statistics::*;
    use distribution::{Univariate, Continuous, Cauchy};

    fn try_create(location: f64, scale: f64) -> Cauchy {
        let n = Cauchy::new(location, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(location: f64, scale: f64) {
        let n = try_create(location, scale);
        assert_eq!(location, n.location());
        assert_eq!(scale, n.scale());
    }

    fn bad_create_case(location: f64, scale: f64) {
        let n = Cauchy::new(location, scale);
        assert!(n.is_err());
    }

    fn test_case<F>(location: f64, scale: f64, expected: f64, eval: F)
        where F: Fn(Cauchy) -> f64
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    #[test]
    fn test_create() {
        create_case(0.0, 0.1);
        create_case(0.0, 1.0);
        create_case(0.0, 10.0);
        create_case(10.0, 11.0);
        create_case(-5.0, 100.0);
        create_case(0.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, 0.0);
    }

    #[test]
    fn test_entropy() {
        test_case(0.0, 2.0, 3.224171427529236102395, |x| x.entropy());
        test_case(0.1, 4.0, 3.917318608089181411812, |x| x.entropy());
        test_case(1.0, 10.0, 4.833609339963336476996, |x| x.entropy());
        test_case(10.0, 11.0, 4.92891951976766133704, |x| x.entropy());
    }

    #[test]
    fn test_mode() {
        test_case(0.0, 2.0, 0.0, |x| x.mode());
        test_case(0.1, 4.0, 0.1, |x| x.mode());
        test_case(1.0, 10.0, 1.0, |x| x.mode());
        test_case(10.0, 11.0, 10.0, |x| x.mode());
        test_case(0.0, f64::INFINITY, 0.0, |x| x.mode());
    }

    #[test]
    fn test_median() {
        test_case(0.0, 2.0, 0.0, |x| x.median());
        test_case(0.1, 4.0, 0.1, |x| x.median());
        test_case(1.0, 10.0, 1.0, |x| x.median());
        test_case(10.0, 11.0, 10.0, |x| x.median());
        test_case(0.0, f64::INFINITY, 0.0, |x| x.median());
    }

    #[test]
    fn test_min_max() {
        test_case(0.0, 1.0, f64::NEG_INFINITY, |x| x.min());
        test_case(0.0, 1.0, f64::INFINITY, |x| x.max());
    }
}