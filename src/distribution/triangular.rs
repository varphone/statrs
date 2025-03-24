use crate::distribution::{Continuous, ContinuousCDF};
use crate::statistics::*;
use std::f64;

/// Implements the
/// [Triangular](https://en.wikipedia.org/wiki/Triangular_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Triangular, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = Triangular::new(0.0, 5.0, 2.5).unwrap();
/// assert_eq!(n.mean().unwrap(), 7.5 / 3.0);
/// assert_eq!(n.pdf(2.5), 5.0 / 12.5);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Triangular {
    min: f64,
    max: f64,
    mode: f64,
}

/// Represents the errors that can occur when creating a [`Triangular`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum TriangularError {
    /// The minimum is NaN or infinite.
    MinInvalid,

    /// The maximum is NaN or infinite.
    MaxInvalid,

    /// The mode is NaN or infinite.
    ModeInvalid,

    /// The mode is less than the minimum or greater than the maximum.
    ModeOutOfRange,

    /// The minimum equals the maximum.
    MinEqualsMax,
}

impl std::fmt::Display for TriangularError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TriangularError::MinInvalid => write!(f, "Minimum is NaN or infinite."),
            TriangularError::MaxInvalid => write!(f, "Maximum is NaN or infinite."),
            TriangularError::ModeInvalid => write!(f, "Mode is NaN or infinite."),
            TriangularError::ModeOutOfRange => {
                write!(f, "Mode is less than minimum or greater than maximum")
            }
            TriangularError::MinEqualsMax => write!(f, "Minimum equals Maximum"),
        }
    }
}

impl std::error::Error for TriangularError {}

impl Triangular {
    /// Constructs a new triangular distribution with a minimum of `min`,
    /// maximum of `max`, and a mode of `mode`.
    ///
    /// # Errors
    ///
    /// Returns an error if `min`, `max`, or `mode` are `NaN` or `±INF`.
    /// Returns an error if `max < mode`, `mode < min`, or `max == min`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Triangular;
    ///
    /// let mut result = Triangular::new(0.0, 5.0, 2.5);
    /// assert!(result.is_ok());
    ///
    /// result = Triangular::new(2.5, 1.5, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(min: f64, max: f64, mode: f64) -> Result<Triangular, TriangularError> {
        if !min.is_finite() {
            return Err(TriangularError::MinInvalid);
        }

        if !max.is_finite() {
            return Err(TriangularError::MaxInvalid);
        }

        if !mode.is_finite() {
            return Err(TriangularError::ModeInvalid);
        }

        if max < mode || mode < min {
            return Err(TriangularError::ModeOutOfRange);
        }

        if min == max {
            return Err(TriangularError::MinEqualsMax);
        }

        Ok(Triangular { min, max, mode })
    }
}

impl std::fmt::Display for Triangular {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Triangular([{},{}], {})", self.min, self.max, self.mode)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<f64> for Triangular {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_unchecked(rng, self.min, self.max, self.mode)
    }
}

impl ContinuousCDF<f64, f64> for Triangular {
    /// Calculates the cumulative distribution function for the triangular
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// if x == min {
    ///     0
    /// } if min < x <= mode {
    ///     (x - min)^2 / ((max - min) * (mode - min))
    /// } else if mode < x < max {
    ///     1 - (max - x)^2 / ((max - min) * (max - mode))
    /// } else {
    ///     1
    /// }
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if x <= a {
            0.0
        } else if x <= c {
            (x - a) * (x - a) / ((b - a) * (c - a))
        } else if x < b {
            1.0 - (b - x) * (b - x) / ((b - a) * (b - c))
        } else {
            1.0
        }
    }

    /// Calculates the survival function for the triangular
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// if x == min {
    ///     1
    /// } if min < x <= mode {
    ///     1 - (x - min)^2 / ((max - min) * (mode - min))
    /// } else if mode < x < max {
    ///     (max - min)^2 / ((max - min) * (max - mode))
    /// } else {
    ///     0
    /// }
    /// ```
    fn sf(&self, x: f64) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if x <= a {
            1.0
        } else if x <= c {
            1.0 - ((x - a) * (x - a) / ((b - a) * (c - a)))
        } else if x < b {
            (b - x) * (b - x) / ((b - a) * (b - c))
        } else {
            0.0
        }
    }

    /// Calculates the inverse cumulative distribution function for the triangular
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// if x < (mode - min) / (max - min) {
    ///     min + ((max - min) * (mode - min) * x)^(1 / 2)
    /// } else {
    ///     max - ((max - min) * (max - mode) * (1 - x))^(1 / 2)
    /// }
    /// ```
    fn inverse_cdf(&self, p: f64) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if !(0.0..=1.0).contains(&p) {
            panic!("x must be in [0, 1]");
        }

        if p < (c - a) / (b - a) {
            a + ((c - a) * (b - a) * p).sqrt()
        } else {
            b - ((b - a) * (b - c) * (1.0 - p)).sqrt()
        }
    }
}

impl Min<f64> for Triangular {
    /// Returns the minimum value in the domain of the
    /// triangular distribution representable by a double precision float
    ///
    /// # Remarks
    ///
    /// The return value is the same min used to construct the distribution
    fn min(&self) -> f64 {
        self.min
    }
}

impl Max<f64> for Triangular {
    /// Returns the maximum value in the domain of the
    /// triangular distribution representable by a double precision float
    ///
    /// # Remarks
    ///
    /// The return value is the same max used to construct the distribution
    fn max(&self) -> f64 {
        self.max
    }
}

impl Distribution<f64> for Triangular {
    /// Returns the mean of the triangular distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (min + max + mode) / 3
    /// ```
    fn mean(&self) -> Option<f64> {
        Some((self.min + self.max + self.mode) / 3.0)
    }

    /// Returns the variance of the triangular distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (min^2 + max^2 + mode^2 - min * max - min * mode - max * mode) / 18
    /// ```
    fn variance(&self) -> Option<f64> {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        Some((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0)
    }

    /// Returns the entropy of the triangular distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 / 2 + ln((max - min) / 2)
    /// ```
    fn entropy(&self) -> Option<f64> {
        Some(0.5 + ((self.max - self.min) / 2.0).ln())
    }

    /// Returns the skewness of the triangular distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (sqrt(2) * (min + max - 2 * mode) * (2 * min - max - mode) * (min - 2 *
    /// max + mode)) /
    /// ( 5 * (min^2 + max^2 + mode^2 - min * max - min * mode - max * mode)^(3
    /// / 2))
    /// ```
    fn skewness(&self) -> Option<f64> {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        let q = f64::consts::SQRT_2 * (a + b - 2.0 * c) * (2.0 * a - b - c) * (a - 2.0 * b + c);
        let d = 5.0 * (a * a + b * b + c * c - a * b - a * c - b * c).powf(3.0 / 2.0);
        Some(q / d)
    }
}

impl Median<f64> for Triangular {
    /// Returns the median of the triangular distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// if mode >= (min + max) / 2 {
    ///     min + sqrt((max - min) * (mode - min) / 2)
    /// } else {
    ///     max - sqrt((max - min) * (max - mode) / 2)
    /// }
    /// ```
    fn median(&self) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if c >= (a + b) / 2.0 {
            a + ((b - a) * (c - a) / 2.0).sqrt()
        } else {
            b - ((b - a) * (b - c) / 2.0).sqrt()
        }
    }
}

impl Mode<Option<f64>> for Triangular {
    /// Returns the mode of the triangular distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// mode
    /// ```
    fn mode(&self) -> Option<f64> {
        Some(self.mode)
    }
}

impl Continuous<f64, f64> for Triangular {
    /// Calculates the probability density function for the triangular
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// if x < min {
    ///     0
    /// } else if min <= x <= mode {
    ///     2 * (x - min) / ((max - min) * (mode - min))
    /// } else if mode < x <= max {
    ///     2 * (max - x) / ((max - min) * (max - mode))
    /// } else {
    ///     0
    /// }
    /// ```
    fn pdf(&self, x: f64) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if a <= x && x <= c {
            2.0 * (x - a) / ((b - a) * (c - a))
        } else if c < x && x <= b {
            2.0 * (b - x) / ((b - a) * (b - c))
        } else {
            0.0
        }
    }

    /// Calculates the log probability density function for the triangular
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln( if x < min {
    ///     0
    /// } else if min <= x <= mode {
    ///     2 * (x - min) / ((max - min) * (mode - min))
    /// } else if mode < x <= max {
    ///     2 * (max - x) / ((max - min) * (max - mode))
    /// } else {
    ///     0
    /// } )
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
fn sample_unchecked<R: ::rand::Rng + ?Sized>(rng: &mut R, min: f64, max: f64, mode: f64) -> f64 {
    let f: f64 = rng.random();
    if f < (mode - min) / (max - min) {
        min + (f * (max - min) * (mode - min)).sqrt()
    } else {
        max - ((1.0 - f) * (max - min) * (max - mode)).sqrt()
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(min: f64, max: f64, mode: f64; Triangular; TriangularError);

    #[test]
    fn test_create() {
        create_ok(-1.0, 1.0, 0.0);
        create_ok(1.0, 2.0, 1.0);
        create_ok(5.0, 25.0, 25.0);
        create_ok(1.0e-5, 1.0e5, 1.0e-3);
        create_ok(0.0, 1.0, 0.9);
        create_ok(-4.0, -0.5, -2.0);
        create_ok(-13.039, 8.42, 1.17);
    }

    #[test]
    fn test_bad_create() {
        let invalid = [
            (0.0, 0.0, 0.0, TriangularError::MinEqualsMax),
            (0.0, 1.0, -0.1, TriangularError::ModeOutOfRange),
            (0.0, 1.0, 1.1, TriangularError::ModeOutOfRange),
            (0.0, -1.0, 0.5, TriangularError::ModeOutOfRange),
            (2.0, 1.0, 1.5, TriangularError::ModeOutOfRange),
            (f64::NAN, 1.0, 0.5, TriangularError::MinInvalid),
            (0.2, f64::NAN, 0.5, TriangularError::MaxInvalid),
            (0.5, 1.0, f64::NAN, TriangularError::ModeInvalid),
            (f64::NAN, f64::NAN, f64::NAN, TriangularError::MinInvalid),
            (f64::NEG_INFINITY, 1.0, 0.5, TriangularError::MinInvalid),
            (0.0, f64::INFINITY, 0.5, TriangularError::MaxInvalid),
        ];

        for (min, max, mode, err) in invalid {
            test_create_err(min, max, mode, err);
        }
    }

    #[test]
    fn test_variance() {
        let variance = |x: Triangular| x.variance().unwrap();
        test_exact(0.0, 1.0, 0.5, 0.75 / 18.0, variance);
        test_exact(0.0, 1.0, 0.75, 0.8125 / 18.0, variance);
        test_exact(-5.0, 8.0, -3.5, 151.75 / 18.0, variance);
        test_exact(-5.0, 8.0, 5.0, 139.0 / 18.0, variance);
        test_exact(-5.0, -3.0, -4.0, 3.0 / 18.0, variance);
        test_exact(15.0, 134.0, 21.0, 13483.0 / 18.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Triangular| x.entropy().unwrap();
        test_absolute(0.0, 1.0, 0.5, -0.1931471805599453094172, 1e-16, entropy);
        test_absolute(0.0, 1.0, 0.75, -0.1931471805599453094172, 1e-16, entropy);
        test_exact(-5.0, 8.0, -3.5, 2.371802176901591426636, entropy);
        test_exact(-5.0, 8.0, 5.0, 2.371802176901591426636, entropy);
        test_exact(-5.0, -3.0, -4.0, 0.5, entropy);
        test_exact(15.0, 134.0, 21.0, 4.585976312551584075938, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Triangular| x.skewness().unwrap();
        test_exact(0.0, 1.0, 0.5, 0.0, skewness);
        test_exact(0.0, 1.0, 0.75, -0.4224039833745502226059, skewness);
        test_exact(-5.0, 8.0, -3.5, 0.5375093589712976359809, skewness);
        test_exact(-5.0, 8.0, 5.0, -0.4445991743012595633537, skewness);
        test_exact(-5.0, -3.0, -4.0, 0.0, skewness);
        test_exact(15.0, 134.0, 21.0, 0.5605920922751860613217, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Triangular| x.mode().unwrap();
        test_exact(0.0, 1.0, 0.5, 0.5, mode);
        test_exact(0.0, 1.0, 0.75, 0.75, mode);
        test_exact(-5.0, 8.0, -3.5, -3.5, mode);
        test_exact(-5.0, 8.0, 5.0, 5.0, mode);
        test_exact(-5.0, -3.0, -4.0, -4.0, mode);
        test_exact(15.0, 134.0, 21.0, 21.0, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: Triangular| x.median();
        test_exact(0.0, 1.0, 0.5, 0.5, median);
        test_exact(0.0, 1.0, 0.75, 0.6123724356957945245493, median);
        test_absolute(-5.0, 8.0, -3.5, -0.6458082328952913226724, 1e-15, median);
        test_absolute(-5.0, 8.0, 5.0, 3.062257748298549652367, 1e-15, median);
        test_exact(-5.0, -3.0, -4.0, -4.0, median);
        test_absolute(15.0, 134.0, 21.0, 52.00304883716712238797, 1e-14, median);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Triangular| x.pdf(arg);
        test_exact(0.0, 1.0, 0.5, 0.0, pdf(-1.0));
        test_exact(0.0, 1.0, 0.5, 0.0, pdf(1.1));
        test_exact(0.0, 1.0, 0.5, 1.0, pdf(0.25));
        test_exact(0.0, 1.0, 0.5, 2.0, pdf(0.5));
        test_exact(0.0, 1.0, 0.5, 1.0, pdf(0.75));
        test_exact(-5.0, 8.0, -3.5, 0.0, pdf(-5.1));
        test_exact(-5.0, 8.0, -3.5, 0.0, pdf(8.1));
        test_exact(-5.0, 8.0, -3.5, 0.1025641025641025641026, pdf(-4.0));
        test_exact(-5.0, 8.0, -3.5, 0.1538461538461538461538, pdf(-3.5));
        test_exact(-5.0, 8.0, -3.5, 0.05351170568561872909699, pdf(4.0));
        test_exact(-5.0, -3.0, -4.0, 0.0, pdf(-5.1));
        test_exact(-5.0, -3.0, -4.0, 0.0, pdf(-2.9));
        test_exact(-5.0, -3.0, -4.0, 0.5, pdf(-4.5));
        test_exact(-5.0, -3.0, -4.0, 1.0, pdf(-4.0));
        test_exact(-5.0, -3.0, -4.0, 0.5, pdf(-3.5));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Triangular| x.ln_pdf(arg);
        test_exact(0.0, 1.0, 0.5, f64::NEG_INFINITY, ln_pdf(-1.0));
        test_exact(0.0, 1.0, 0.5, f64::NEG_INFINITY, ln_pdf(1.1));
        test_exact(0.0, 1.0, 0.5, 0.0, ln_pdf(0.25));
        test_exact(0.0, 1.0, 0.5, 2f64.ln(), ln_pdf(0.5));
        test_exact(0.0, 1.0, 0.5, 0.0, ln_pdf(0.75));
        test_exact(-5.0, 8.0, -3.5, f64::NEG_INFINITY, ln_pdf(-5.1));
        test_exact(-5.0, 8.0, -3.5, f64::NEG_INFINITY, ln_pdf(8.1));
        test_exact(-5.0, 8.0, -3.5, 0.1025641025641025641026f64.ln(), ln_pdf(-4.0));
        test_exact(-5.0, 8.0, -3.5, 0.1538461538461538461538f64.ln(), ln_pdf(-3.5));
        test_exact(-5.0, 8.0, -3.5, 0.05351170568561872909699f64.ln(), ln_pdf(4.0));
        test_exact(-5.0, -3.0, -4.0, f64::NEG_INFINITY, ln_pdf(-5.1));
        test_exact(-5.0, -3.0, -4.0, f64::NEG_INFINITY, ln_pdf(-2.9));
        test_exact(-5.0, -3.0, -4.0, 0.5f64.ln(), ln_pdf(-4.5));
        test_exact(-5.0, -3.0, -4.0, 0.0, ln_pdf(-4.0));
        test_exact(-5.0, -3.0, -4.0, 0.5f64.ln(), ln_pdf(-3.5));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Triangular| x.cdf(arg);
        test_exact(0.0, 1.0, 0.5, 0.125, cdf(0.25));
        test_exact(0.0, 1.0, 0.5, 0.5, cdf(0.5));
        test_exact(0.0, 1.0, 0.5, 0.875, cdf(0.75));
        test_exact(-5.0, 8.0, -3.5, 0.05128205128205128205128, cdf(-4.0));
        test_exact(-5.0, 8.0, -3.5, 0.1153846153846153846154, cdf(-3.5));
        test_exact(-5.0, 8.0, -3.5, 0.892976588628762541806, cdf(4.0));
        test_exact(-5.0, -3.0, -4.0, 0.125, cdf(-4.5));
        test_exact(-5.0, -3.0, -4.0, 0.5, cdf(-4.0));
        test_exact(-5.0, -3.0, -4.0, 0.875, cdf(-3.5));
    }

    #[test]
    fn test_cdf_lower_bound() {
        let cdf = |arg: f64| move |x: Triangular| x.cdf(arg);
        test_exact(0.0, 3.0, 1.5, 0.0, cdf(-1.0));
    }

    #[test]
    fn test_cdf_upper_bound() {
        let cdf = |arg: f64| move |x: Triangular| x.cdf(arg);
        test_exact(0.0, 3.0, 1.5, 1.0, cdf(5.0));
    }


    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Triangular| x.sf(arg);
        test_exact(0.0, 1.0, 0.5, 0.875, sf(0.25));
        test_exact(0.0, 1.0, 0.5, 0.5, sf(0.5));
        test_exact(0.0, 1.0, 0.5, 0.125, sf(0.75));
        test_exact(-5.0, 8.0, -3.5, 0.9487179487179487, sf(-4.0));
        test_exact(-5.0, 8.0, -3.5, 0.8846153846153846, sf(-3.5));
        test_exact(-5.0, 8.0, -3.5, 0.10702341137123746, sf(4.0));
        test_exact(-5.0, -3.0, -4.0, 0.875, sf(-4.5));
        test_exact(-5.0, -3.0, -4.0, 0.5, sf(-4.0));
        test_exact(-5.0, -3.0, -4.0, 0.125, sf(-3.5));
    }

    #[test]
    fn test_sf_lower_bound() {
        let sf = |arg: f64| move |x: Triangular| x.sf(arg);
        test_exact(0.0, 3.0, 1.5, 1.0, sf(-1.0));
    }

    #[test]
    fn test_sf_upper_bound() {
        let sf = |arg: f64| move |x: Triangular| x.sf(arg);
        test_exact(0.0, 3.0, 1.5, 0.0, sf(5.0));
    }

    #[test]
    fn test_inverse_cdf() {
        let func = |arg: f64| move |x: Triangular| x.inverse_cdf(x.cdf(arg));
        test_absolute(0.0, 1.0, 0.5, 0.25, 1e-15, func(0.25));
        test_absolute(0.0, 1.0, 0.5, 0.5, 1e-15, func(0.5));
        test_absolute(0.0, 1.0, 0.5, 0.75, 1e-15, func(0.75));
        test_absolute(-5.0, 8.0, -3.5, -4.0, 1e-15, func(-4.0));
        test_absolute(-5.0, 8.0, -3.5, -3.5, 1e-15, func(-3.5));
        test_absolute(-5.0, 8.0, -3.5, 4.0, 1e-15, func(4.0));
        test_absolute(-5.0, -3.0, -4.0, -4.5, 1e-15, func(-4.5));
        test_absolute(-5.0, -3.0, -4.0, -4.0, 1e-15, func(-4.0));
        test_absolute(-5.0, -3.0, -4.0, -3.5, 1e-15, func(-3.5));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&create_ok(-5.0, 5.0, 0.0), -5.0, 5.0);
        test::check_continuous_distribution(&create_ok(-15.0, -2.0, -3.0), -15.0, -2.0);
    }
}
