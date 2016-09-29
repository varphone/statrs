use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use error::StatsError;
use function::{beta, gamma};
use result::Result;
use {Min, Max, Mean, Variance};
use super::*;

/// Implements the [Student's T](https://en.wikipedia.org/wiki/Student%27s_t-distribution) distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{StudentsT, Continuous};
/// use statrs::Mean;
/// use statrs::prec;
///
/// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
/// assert_eq!(n.mean(), 0.0);
/// assert!(prec::almost_eq(n.pdf(0.0), 0.353553390593274, 1e-15));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StudentsT {
    location: f64,
    scale: f64,
    freedom: f64,
}

impl StudentsT {
    /// Constructs a new student's t-distribution with location `location`, scale `scale`,
    /// and `freedom` freedom.
    ///
    /// # Errors
    ///
    /// Returns an error if any of `location`, `scale`, or `freedom` are `NaN`.
    /// Returns an error if `scale <= 0.0` or `freedom <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let mut result = StudentsT::new(0.0, 1.0, 2.0);
    /// assert!(result.is_ok());
    ///
    /// result = StudentsT::new(0.0, 0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64, freedom: f64) -> Result<StudentsT> {
        let is_nan = location.is_nan() || scale.is_nan() || freedom.is_nan();
        if is_nan || scale <= 0.0 || freedom <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(StudentsT {
                location: location,
                scale: scale,
                freedom: freedom,
            })
        }
    }

    /// Returns the location of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the freedom of the student's t-distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::StudentsT;
    ///
    /// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// assert_eq!(n.freedom(), 2.0);
    /// ```
    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl Sample<f64> for StudentsT {
    /// Generate a random sample from a student's t-distribution
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for StudentsT {
    /// Generate a random independent sample from a student's t-distribution
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for StudentsT {
    /// Generate a random sample from a student's t-distribution using
    /// `r` as the source of randomness. The implementation is based
    /// on method 2, section 5 in chapter 9 of L. Devroye's
    /// <i>"Non-Uniform Random Variate Generation"</i>
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{StudentsT, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        let gamma = super::gamma::sample_unchecked(r, 0.5 * self.freedom, 0.5);
        super::normal::sample_unchecked(r,
                                        self.location,
                                        self.scale * (self.freedom / gamma).sqrt())
    }
}

impl Univariate<f64, f64> for StudentsT {
    /// Calculates the cumulative distribution function for the student's t-distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < μ {
    ///     (1 / 2) * I(t, v / 2, 1 / 2)
    /// } else {
    ///     1 - (1 / 2) * I(t, v / 2, 1 / 2)
    /// }
    /// ```
    ///
    /// where `t = v / (v + k^2)`, `k = (x - μ) / σ`, `μ` is the location,
    /// `σ` is the scale, `v` is the freedom, and `I` is the regularized incomplete
    /// beta function
    fn cdf(&self, x: f64) -> f64 {
        if self.freedom == f64::INFINITY {
            super::normal::cdf_unchecked(x, self.location, self.scale)
        } else {
            let k = (x - self.location) / self.scale;
            let h = self.freedom / (self.freedom + k * k);
            let ib = 0.5 * beta::beta_reg(self.freedom / 2.0, 0.5, h);
            if x <= self.location { ib } else { 1.0 - ib }
        }
    }
}

impl Min<f64> for StudentsT {
    /// Returns the minimum value in the domain of the student's t-distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// -INF
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for StudentsT {
    /// Returns the maximum value in the domain of the student's t-distribution
    /// representable by a double precision float
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

impl Mean<f64> for StudentsT {
    /// Returns the mean of the student's t-distribution
    ///
    /// # Panics
    ///
    /// If `freedom <= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mean(&self) -> f64 {
        assert!(self.freedom > 1.0,
                format!("{}", StatsError::ArgGt("freedom", 1.0)));
        self.location
    }
}

impl Variance<f64> for StudentsT {
    /// Returns the variance of the student's t-distribution
    ///
    /// # Panics
    ///
    /// If `freedom <= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if v == INF {
    ///     σ^2
    /// } else if freedom > 2.0 {
    ///     v * σ^2 / (v - 2)
    /// } else {
    ///     INF
    /// }
    /// ```
    ///
    /// where `σ` is the scale and `v` is the freedom
    fn variance(&self) -> f64 {
        assert!(self.freedom > 1.0,
                format!("{}", StatsError::ArgGt("freedom", 1.0)));
        if self.freedom == f64::INFINITY {
            self.scale * self.scale
        } else if self.freedom > 2.0 {
            self.freedom * self.scale * self.scale / (self.freedom - 2.0)
        } else {
            f64::INFINITY
        }
    }

    /// Returns the standard deviation of the student's t-distribution
    ///
    /// # Panics
    ///
    /// If `freedom <= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// let variance = if v == INF {
    ///     σ^2
    /// } else if freedom > 2.0 {
    ///     v * σ^2 / (v - 2)
    /// } else {
    ///     INF
    /// }
    /// sqrt(variance)
    /// ```
    ///
    /// where `σ` is the scale and `v` is the freedom
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Entropy<f64> for StudentsT {
    /// Returns the entropy for the student's t-distribution
    ///
    /// # Panics
    ///
    /// If `location != 0.0 && scale != 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (v + 1) / 2 * (ψ((v + 1) / 2) - ψ(v / 2)) + ln(sqrt(v) * B(v / 2, 1 / 2))
    /// ```
    ///
    /// where `v` is the freedom, `ψ` is the digamma function, and `B` is the beta function
    fn entropy(&self) -> f64 {
        assert!(self.location == 0.0 && self.scale == 1.0,
                "Cannot calculate entropy for StudentsT distribution where location is not 0 and \
                 scale is not 1");

        (self.freedom + 1.0) / 2.0 *
        (gamma::digamma((self.freedom + 1.0) / 2.0) - gamma::digamma(self.freedom / 2.0)) +
        (self.freedom.sqrt() * beta::beta(self.freedom / 2.0, 0.5)).ln()
    }
}

impl Skewness<f64, f64> for StudentsT {
    /// Returns the skewness of the student's t-distribution
    ///
    /// # Panics
    ///
    /// If `x <= 3.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn skewness(&self) -> f64 {
        assert!(self.freedom > 3.0,
                format!("{}", StatsError::ArgGt("freedom", 3.0)));
        0.0
    }
}

impl Median<f64> for StudentsT {
    /// Returns the median of the student's t-distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn median(&self) -> f64 {
        self.location
    }
}

impl Mode<f64, f64> for StudentsT {
    /// Returns the mode of the student's t-distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mode(&self) -> f64 {
        self.location
    }
}

impl Continuous<f64, f64> for StudentsT {
    /// Calculates the probability density function for the student's t-distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Γ((v + 1) / 2) / (sqrt(vπ) * Γ(v / 2) * σ) * (1 + k^2 / v)^(-1 / 2 * (v + 1))
    /// ```
    ///
    /// where `k = (x - μ) / σ`, `μ` is the location, `σ` is the scale, `v` is the freedom,
    /// and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        if self.freedom >= 1e8 {
            super::normal::pdf_unchecked(x, self.location, self.scale)
        } else {
            let d = (x - self.location) / self.scale;
            (gamma::ln_gamma((self.freedom + 1.0) / 2.0) - gamma::ln_gamma(self.freedom / 2.0))
                .exp() *
            (1.0 + d * d / self.freedom).powf(-0.5 * (self.freedom + 1.0)) /
            (self.freedom * f64::consts::PI).sqrt() / self.scale
        }
    }

    /// Calculates the log probability density function for the student's t-distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(Γ((v + 1) / 2) / (sqrt(vπ) * Γ(v / 2) * σ) * (1 + k^2 / v)^(-1 / 2 * (v + 1)))
    /// ```
    ///
    /// where `k = (x - μ) / σ`, `μ` is the location, `σ` is the scale, `v` is the freedom,
    /// and `Γ` is the gamma function
    fn ln_pdf(&self, x: f64) -> f64 {
        if self.freedom >= 1e8 {
            super::normal::ln_pdf_unchecked(x, self.location, self.scale)
        } else {
            let d = (x - self.location) / self.scale;
            gamma::ln_gamma((self.freedom + 1.0) / 2.0) -
            0.5 * ((self.freedom + 1.0) * (1.0 + d * d / self.freedom).ln()) -
            gamma::ln_gamma(self.freedom / 2.0) -
            0.5 * (self.freedom * f64::consts::PI).ln() - self.scale.ln()
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use std::panic;
    use distribution::*;
    use {Min, Max, Mean, Variance};

    fn try_create(location: f64, scale: f64, freedom: f64) -> StudentsT {
        let n = StudentsT::new(location, scale, freedom);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(location: f64, scale: f64, freedom: f64) {
        let n = try_create(location, scale, freedom);
        assert_eq!(n.location(), location);
        assert_eq!(n.scale(), scale);
        assert_eq!(n.freedom(), freedom);
    }

    fn bad_create_case(location: f64, scale: f64, freedom: f64) {
        let n = StudentsT::new(location, scale, freedom);
        assert!(n.is_err());
    }

    fn get_value<F>(location: f64, scale: f64, freedom: f64, eval: F) -> f64
        where F: Fn(StudentsT) -> f64
    {
        let n = try_create(location, scale, freedom);
        eval(n)
    }

    fn test_case<F>(location: f64, scale: f64, freedom: f64, expected: f64, eval: F)
        where F: Fn(StudentsT) -> f64
    {
        let x = get_value(location, scale, freedom, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(location: f64, scale: f64, freedom: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(StudentsT) -> f64
    {
        let x = get_value(location, scale, freedom, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_panic<F>(location: f64, scale: f64, freedom: f64, eval: F)
        where F : Fn(StudentsT) -> f64,
              F : panic::UnwindSafe
    {
        let result = panic::catch_unwind(|| {
            get_value(location, scale, freedom, eval)
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_create() {
        create_case(0.0, 0.1, 1.0);
        create_case(0.0, 1.0, 1.0);
        create_case(-5.0, 1.0, 3.0);
        create_case(10.0, 10.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0, 1.0);
        bad_create_case(0.0, f64::NAN, 1.0);
        bad_create_case(0.0, 1.0, f64::NAN);
        bad_create_case(0.0, -10.0, 1.0);
        bad_create_case(0.0, 10.0, -1.0);
    }

    #[test]
    fn test_mean() {
        test_panic(0.0, 1.0, 1.0, |x| x.mean());
        test_panic(0.0, 0.1, 1.0, |x| x.mean());
        test_case(0.0, 1.0, 3.0, 0.0, |x| x.mean());
        test_panic(0.0, 10.0, 1.0, |x| x.mean());
        test_case(0.0, 10.0, 2.0, 0.0, |x| x.mean());
        test_case(0.0, 10.0, f64::INFINITY, 0.0, |x| x.mean());
        test_panic(10.0, 1.0, 1.0, |x| x.mean());
        test_case(-5.0, 100.0, 1.5, -5.0, |x| x.mean());
        test_panic(0.0, f64::INFINITY, 1.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_panic(0.0, 1.0, 1.0, |x| x.variance());
        test_panic(0.0, 0.1, 1.0, |x| x.variance());
        test_case(0.0, 1.0, 3.0, 3.0, |x| x.variance());
        test_panic(0.0, 10.0, 1.0, |x| x.variance());
        test_case(0.0, 10.0, 2.0, f64::INFINITY, |x| x.variance());
        test_case(0.0, 10.0, 2.5, 500.0, |x| x.variance());
        test_panic(10.0, 1.0, 1.0, |x| x.variance());
        test_case(10.0, 1.0, 2.5, 5.0, |x| x.variance());
        test_case(-5.0, 100.0, 1.5, f64::INFINITY, |x| x.variance());
        test_panic(0.0, f64::INFINITY, 1.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_panic(0.0, 1.0, 1.0, |x| x.std_dev());
        test_panic(0.0, 0.1, 1.0, |x| x.std_dev());
        test_case(0.0, 1.0, 3.0, 1.7320508075688772935274463415059, |x| x.std_dev());
        test_panic(0.0, 10.0, 1.0, |x| x.std_dev());
        test_case(0.0, 10.0, 2.0, f64::INFINITY, |x| x.std_dev());
        test_case(0.0, 10.0, 2.5, 22.360679774997896964091736687313, |x| x.std_dev());
        test_case(0.0, 10.0, f64::INFINITY, 10.0, |x| x.std_dev());
        test_panic(10.0, 1.0, 1.0, |x| x.std_dev());
        test_case(10.0, 1.0, 2.5, 2.2360679774997896964091736687313, |x| x.std_dev());
        test_case(-5.0, 100.0, 1.5, f64::INFINITY, |x| x.std_dev());
        test_panic(0.0, f64::INFINITY, 1.0, |x| x.std_dev());
    }

    #[test]
    fn test_mode() {
        test_case(0.0, 1.0, 1.0, 0.0, |x| x.mode());
        test_case(0.0, 0.1, 1.0, 0.0, |x| x.mode());
        test_case(0.0, 1.0, 3.0, 0.0, |x| x.mode());
        test_case(0.0, 10.0, 1.0, 0.0, |x| x.mode());
        test_case(0.0, 10.0, 2.0, 0.0, |x| x.mode());
        test_case(0.0, 10.0, 2.5, 0.0, |x| x.mode());
        test_case(0.0, 10.0, f64::INFINITY, 0.0, |x| x.mode());
        test_case(10.0, 1.0, 1.0, 10.0, |x| x.mode());
        test_case(10.0, 1.0, 2.5, 10.0, |x| x.mode());
        test_case(-5.0, 100.0, 1.5, -5.0, |x| x.mode());
        test_case(0.0, f64::INFINITY, 1.0, 0.0, |x| x.mode());
    }

    #[test]
    fn test_median() {
        test_case(0.0, 1.0, 1.0, 0.0, |x| x.median());
        test_case(0.0, 0.1, 1.0, 0.0, |x| x.median());
        test_case(0.0, 1.0, 3.0, 0.0, |x| x.median());
        test_case(0.0, 10.0, 1.0, 0.0, |x| x.median());
        test_case(0.0, 10.0, 2.0, 0.0, |x| x.median());
        test_case(0.0, 10.0, 2.5, 0.0, |x| x.median());
        test_case(0.0, 10.0, f64::INFINITY, 0.0, |x| x.median());
        test_case(10.0, 1.0, 1.0, 10.0, |x| x.median());
        test_case(10.0, 1.0, 2.5, 10.0, |x| x.median());
        test_case(-5.0, 100.0, 1.5, -5.0, |x| x.median());
        test_case(0.0, f64::INFINITY, 1.0, 0.0, |x| x.median());
    }

    #[test]
    fn test_min_max() {
        test_case(0.0, 1.0, 1.0, f64::NEG_INFINITY, |x| x.min());
        test_case(2.5, 100.0, 1.5, f64::NEG_INFINITY, |x| x.min());
        test_case(10.0, f64::INFINITY, 3.5, f64::NEG_INFINITY, |x| x.min());
        test_case(0.0, 1.0, 1.0, f64::INFINITY, |x| x.max());
        test_case(2.5, 100.0, 1.5, f64::INFINITY, |x| x.max());
        test_case(10.0, f64::INFINITY, 5.5, f64::INFINITY, |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_almost(0.0, 1.0, 1.0, 0.318309886183791, 1e-15, |x| x.pdf(0.0));
        test_almost(0.0, 1.0, 1.0, 0.159154943091895, 1e-15, |x| x.pdf(1.0));
        test_almost(0.0, 1.0, 1.0, 0.159154943091895, 1e-15, |x| x.pdf(-1.0));
        test_almost(0.0, 1.0, 1.0, 0.063661977236758, 1e-15, |x| x.pdf(2.0));
        test_almost(0.0, 1.0, 1.0, 0.063661977236758, 1e-15, |x| x.pdf(-2.0));
        test_almost(0.0, 1.0, 2.0, 0.353553390593274, 1e-15, |x| x.pdf(0.0));
        test_almost(0.0, 1.0, 2.0, 0.192450089729875, 1e-15, |x| x.pdf(1.0));
        test_almost(0.0, 1.0, 2.0, 0.192450089729875, 1e-15, |x| x.pdf(-1.0));
        test_almost(0.0, 1.0, 2.0, 0.068041381743977, 1e-15, |x| x.pdf(2.0));
        test_almost(0.0, 1.0, 2.0, 0.068041381743977, 1e-15, |x| x.pdf(-2.0));
        test_almost(0.0, 1.0, f64::INFINITY, 0.398942280401433, 1e-15, |x| x.pdf(0.0));
        test_almost(0.0, 1.0, f64::INFINITY, 0.241970724519143, 1e-15, |x| x.pdf(1.0));
        test_almost(0.0, 1.0, f64::INFINITY, 0.053990966513188, 1e-15, |x| x.pdf(2.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_almost(0.0, 1.0, 1.0, -1.144729885849399, 1e-14, |x| x.ln_pdf(0.0));
        test_almost(0.0, 1.0, 1.0, -1.837877066409348, 1e-14, |x| x.ln_pdf(1.0));
        test_almost(0.0, 1.0, 1.0, -1.837877066409348, 1e-14, |x| x.ln_pdf(-1.0));
        test_almost(0.0, 1.0, 1.0, -2.754167798283503, 1e-14, |x| x.ln_pdf(2.0));
        test_almost(0.0, 1.0, 1.0, -2.754167798283503, 1e-14, |x| x.ln_pdf(-2.0));
        test_almost(0.0, 1.0, 2.0, -1.039720770839917, 1e-14, |x| x.ln_pdf(0.0));
        test_almost(0.0, 1.0, 2.0, -1.647918433002166, 1e-14, |x| x.ln_pdf(1.0));
        test_almost(0.0, 1.0, 2.0, -1.647918433002166, 1e-14, |x| x.ln_pdf(-1.0));
        test_almost(0.0, 1.0, 2.0, -2.687639203842085, 1e-14, |x| x.ln_pdf(2.0));
        test_almost(0.0, 1.0, 2.0, -2.687639203842085, 1e-14, |x| x.ln_pdf(-2.0));
        test_almost(0.0, 1.0, f64::INFINITY, -0.918938533204672, 1e-14, |x| x.ln_pdf(0.0));
        test_almost(0.0, 1.0, f64::INFINITY, -1.418938533204674, 1e-14, |x| x.ln_pdf(1.0));
        test_almost(0.0, 1.0, f64::INFINITY, -2.918938533204674, 1e-14, |x| x.ln_pdf(2.0));
    }

    #[test]
    fn test_cdf() {
        test_case(0.0, 1.0, 1.0, 0.5, |x| x.cdf(0.0));
        test_almost(0.0, 1.0, 1.0, 0.75, 1e-15, |x| x.cdf(1.0));
        test_almost(0.0, 1.0, 1.0, 0.25, 1e-15, |x| x.cdf(-1.0));
        test_almost(0.0, 1.0, 1.0, 0.852416382349567, 1e-15, |x| x.cdf(2.0));
        test_almost(0.0, 1.0, 1.0, 0.147583617650433, 1e-15, |x| x.cdf(-2.0));
        test_case(0.0, 1.0, 2.0, 0.5, |x| x.cdf(0.0));
        test_almost(0.0, 1.0, 2.0, 0.788675134594813, 1e-15, |x| x.cdf(1.0));
        test_almost(0.0, 1.0, 2.0, 0.211324865405187, 1e-15, |x| x.cdf(-1.0));
        test_almost(0.0, 1.0, 2.0, 0.908248290463863, 1e-15, |x| x.cdf(2.0));
        test_almost(0.0, 1.0, 2.0, 0.091751709536137, 1e-15, |x| x.cdf(-2.0));
        test_case(0.0, 1.0, f64::INFINITY, 0.5, |x| x.cdf(0.0));

// TODO: these are curiously low accuracy and should be re-examined
        test_almost(0.0, 1.0, f64::INFINITY, 0.841344746068543, 1e-10, |x| x.cdf(1.0));
        test_almost(0.0, 1.0, f64::INFINITY, 0.977249868051821, 1e-11, |x| x.cdf(2.0));
    }
}
