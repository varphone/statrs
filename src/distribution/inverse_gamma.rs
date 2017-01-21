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

impl Mean<f64> for InverseGamma {
    /// Returns the mean of the inverse distribution
    ///
    /// # Panics
    ///
    /// If `shape <= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// β / (α - 1)
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn mean(&self) -> f64 {
        assert!(self.shape > 1.0,
                format!("{}", StatsError::ArgGt("shape", 1.0)));
        self.rate / (self.shape - 1.0)
    }
}

impl Variance<f64> for InverseGamma {
    /// Returns the variance of the inverse gamma distribution
    ///
    /// # Panics
    ///
    /// If `shape <= 2.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// β^2 / ((α - 1)^2 * (α - 2))
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn variance(&self) -> f64 {
        assert!(self.shape > 2.0,
                format!("{}", StatsError::ArgGt("shape", 2.0)));
        self.rate * self.rate / ((self.shape - 1.0) * (self.shape - 1.0) * (self.shape - 2.0))
    }

    /// Returns the standard deviation of the inverse gamma distribution
    ///
    /// # Panics
    ///
    /// If `shape <= 2.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(β^2 / ((α - 1)^2 * (α - 2)))
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Entropy<f64> for InverseGamma {
    /// Returns the entropy of the inverse gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// α + ln(β * Γ(α)) - (1 + α) * ψ(α)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> f64 {
        self.shape + self.rate.ln() + gamma::ln_gamma(self.shape) -
        (1.0 + self.shape) * gamma::digamma(self.shape)
    }
}

impl Skewness<f64> for InverseGamma {
    /// Returns the skewness of the inverse gamma distribution
    ///
    /// # Panics
    ///
    /// If `shape <= 3`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 4 * sqrt(α - 2) / (α - 3)
    /// ```
    ///
    /// where `α` is the shape
    fn skewness(&self) -> f64 {
        assert!(self.shape > 3.0,
                format!("{}", StatsError::ArgGt("shape", 3.0)));
        4.0 * (self.shape - 2.0).sqrt() / (self.shape - 3.0)
    }
}

impl Mode<f64> for InverseGamma {
    /// Returns the mode of the inverse gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// β / (α + 1)
    /// ```
    ///
    /// /// where `α` is the shape and `β` is the rate
    fn mode(&self) -> f64 {
        self.rate / (self.shape + 1.0)
    }
}

impl Continuous<f64, f64> for InverseGamma {
    /// Calculates the probability density function for the
    /// inverse gamma distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x <= 0.0`
    ///
    /// # Remarks
    ///
    /// Returns `NAN` if `rate` is `INF`, returns `0.0`
    /// if `shape` is `INF`, returns `NAN` if both are `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (β^α / Γ(α)) * x^(-α - 1) * e^(-β / x)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        assert!(x > 0.0, format!("{}", StatsError::ArgMustBePositive("x")));
        if self.shape == 1.0 {
            self.rate / (x * x) * (-self.rate / x).exp()
        } else {
            self.rate.powf(self.shape) * x.powf(-self.shape - 1.0) * (-self.rate / x).exp() /
            gamma::gamma(self.shape)
        }
    }

    /// Calculates the probability density function for the
    /// inverse gamma distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x <= 0.0`
    ///
    /// # Remarks
    ///
    /// Returns `NAN` if `rate` is `INF`, returns `-INF`
    /// if `shape` is `INF`, returns `NAN` if both are `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((β^α / Γ(α)) * x^(-α - 1) * e^(-β / x))
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
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
    #[should_panic]
    fn test_cdf_panics() {
        get_value(0.1, 0.1, |x| x.cdf(-1.0));
    }

    #[test]
    fn test_min_max() {
        test_case(1.0, 1.0, 0.0, |x| x.min());
        test_case(1.0, 1.0, f64::INFINITY, |x| x.max());
    }

    #[test]
    fn test_mean() {
        test_almost(1.1, 0.1, 1.0, 1e-14, |x| x.mean());
        test_almost(1.1, 1.0, 10.0, 1e-14, |x| x.mean());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.mean());
    }

    #[test]
    #[should_panic]
    fn test_mean_panics() {
        get_value(0.1, 0.1, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_almost(2.1, 0.1, 0.08264462809917355371901, 1e-15, |x| x.variance());
        test_almost(2.1, 1.0, 8.264462809917355371901, 1e-13, |x| x.variance());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.variance());
    }

    #[test]
    #[should_panic]
    fn test_variance_panics() {
        get_value(0.1, 0.1, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_almost(2.1, 0.1, 0.2874797872880344847272, 1e-15, |x| x.std_dev());
        test_almost(2.1, 1.0, 2.874797872880344847272, 1e-14, |x| x.std_dev());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.std_dev());
    }

    #[test]
    #[should_panic]
    fn test_std_dev_panics() {
        get_value(0.1, 0.1, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(0.1, 0.1, 11.51625799319234475054, 1e-14, |x| x.entropy());
        test_almost(1.0, 1.0, 2.154431329803065721213, 1e-14, |x| x.entropy());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_almost(3.1, 0.1, 41.95235392680606187966, 1e-13, |x| x.skewness());
        test_almost(3.1, 1.0, 41.95235392680606187966, 1e-13, |x| x.skewness());
        test_case(5.0, 0.1, 3.464101615137754587055, |x| x.skewness());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.skewness());
    }

    #[test]
    #[should_panic]
    fn test_skewness_panics() {
        get_value(0.1, 0.1, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(0.1, 0.1, 0.09090909090909090909091, |x| x.mode());
        test_case(1.0, 1.0, 0.5, |x| x.mode());
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.mode());
    }

    #[test]
    fn test_pdf() {
        test_almost(0.1, 0.1, 0.0628591853882328004197, 1e-15, |x| x.pdf(1.2));
        test_almost(0.1, 1.0, 0.0297426109178248997426, 1e-15, |x| x.pdf(2.0));
        test_is_nan(0.1 ,f64::INFINITY, |x| x.pdf(1.1));
        test_case(1.0, 0.1, 0.04157808822362745501024, |x| x.pdf(1.5));
        test_case(1.0, 1.0, 0.3018043114632487660842, |x| x.pdf(1.2));
        test_is_nan(1.0, f64::INFINITY, |x| x.pdf(1.5));
        test_case(f64::INFINITY, 0.1, 0.0, |x| x.pdf(5.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(2.5));
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.pdf(1.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_almost(0.1, 0.1, 0.0628591853882328004197f64.ln(), 1e-15, |x| x.ln_pdf(1.2));
        test_almost(0.1, 1.0, 0.0297426109178248997426f64.ln(), 1e-15, |x| x.ln_pdf(2.0));
        test_is_nan(0.1 ,f64::INFINITY, |x| x.ln_pdf(1.1));
        test_case(1.0, 0.1, 0.04157808822362745501024f64.ln(), |x| x.ln_pdf(1.5));
        test_case(1.0, 1.0, 0.3018043114632487660842f64.ln(), |x| x.ln_pdf(1.2));
        test_is_nan(1.0, f64::INFINITY, |x| x.ln_pdf(1.5));
        test_case(f64::INFINITY, 0.1, f64::NEG_INFINITY, |x| x.ln_pdf(5.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(2.5));
        test_is_nan(f64::INFINITY, f64::INFINITY, |x| x.ln_pdf(1.0));
    }
}