//! Defines common interfaces for interacting with statistical distributions and provides
//! concrete implementations for a variety of distributions.

use rand::Rng;
use super::result::Result;

pub use self::bernoulli::Bernoulli;
pub use self::binomial::Binomial;
pub use self::chi::Chi;
pub use self::chi_squared::ChiSquared;
pub use self::discrete_uniform::DiscreteUniform;
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::log_normal::LogNormal;
pub use self::normal::Normal;
pub use self::poisson::Poisson;
pub use self::student_t::StudentT;
pub use self::triangular::Triangular;
pub use self::uniform::Uniform;
pub use self::weibull::Weibull;

mod bernoulli;
mod binomial;
mod chi;
mod chi_squared;
mod discrete_uniform;
mod exponential;
mod gamma;
mod log_normal;
mod normal;
mod poisson;
mod student_t;
mod triangular;
mod uniform;
mod weibull;

/// The `Distribution` trait is used to specify an interface
/// for sampling distributions
///
/// # Examples  
///
/// A trivial implementation that just samples from the supplied
/// random number generator
///
/// ```
/// # extern crate rand;
/// # extern crate statrs;
///
/// use rand::Rng;
/// use statrs::distribution::Distribution;
///
/// struct Foo;
///
/// impl Distribution for Foo {
///     fn sample<R: Rng>(&self, r: &mut R) -> f64 {
///         r.next_f64()
///     }
/// }
///
/// # fn main() { }
/// ```
pub trait Distribution {
    /// Draws a random sample using the supplied random number generator
    fn sample<R: Rng>(&self, r: &mut R) -> f64;
}

/// The `Univariate` trait extends the `Distribution` 
/// trait provides an interface for interacting with
/// univariate statistical distributions.
///
/// # Remarks
///
/// All methods provided by the `Univariate` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution. The `CheckedUnivariate`
/// trait provides a panic-safe interface for univariate distributions
pub trait Univariate : Distribution {
    /// Returns the mean for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.mean());
    /// ```
    fn mean(&self) -> f64;

    /// Returns the variance for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0 / 12.0, n.variance());
    /// ```
    fn variance(&self) -> f64;

    /// Returns the standard deviation for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    /// 
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!((1f64 / 12f64).sqrt(), n.std_dev());
    /// ```
    fn std_dev(&self) -> f64;

    /// Returns the entropy for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.entropy());
    /// ```
    fn entropy(&self) -> f64;

    /// Returns the skewness for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.skewness());
    /// ```
    fn skewness(&self) -> f64;

    /// Returns the median for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.median());
    /// ```
    fn median(&self) -> f64;

    /// Returns the cumulative distribution function calculated
    /// at `x` for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Univariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.cdf(0.5));
    /// ```
    fn cdf(&self, x: f64) -> f64;
}

/// The `CheckedUnivariate` trait extends the `Distribution` trait and
/// provides a checked interface for interacting with univariate statistical
/// distributions. This means implementors should return an `StatsError` instead of
/// panicking on an invalid state or input.
pub trait CheckedUnivariate : Distribution {
    /// Returns the mean for a given distribution or a `StatsError`
    /// depending on the implementor. This method should not panic
    /// but return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedUnivariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.checked_mean().unwrap());
    /// ```
    fn checked_mean(&self) -> Result<f64>;

    /// Returns the variance for a given distribution or a `StatsError`
    /// depending on the implementor. This method should not panic
    /// but return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedUnivariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0 / 12.0, n.checked_variance().unwrap());
    /// ```
    fn checked_variance(&self) -> Result<f64>;

    /// Returns the standard deviation for a given distribution or
    /// a `StatsError` depending on the implementor. This method should
    /// not panic but return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedUnivariate, Uniform};
    /// 
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!((1f64 / 12f64).sqrt(), n.checked_std_dev().unwrap());
    /// ```
    fn checked_std_dev(&self) -> Result<f64>;

    /// Returns the entropy for a given distribution or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedUnivariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.checked_entropy().unwrap());
    /// ```
    fn checked_entropy(&self) -> Result<f64>;

    /// Returns the skewness for a given distribution or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedUnivariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.checked_skewness().unwrap());
    /// ```
    fn checked_skewness(&self) -> Result<f64>;

    /// Returns the median for a given distribution or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedUnivariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.checked_median().unwrap());
    /// ```
    fn checked_median(&self) -> Result<f64>;

    /// Returns the cumulative distribution function calculated
    /// at `x` for a given distribution or a `StatsError` depending
    /// on the implementor. This method should not panic but return a
    /// `StatsError` instead. `StatsError`s should be returned for argument
    /// bound violations.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedUnivariate, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.checked_cdf(0.5).unwrap());
    /// ```
    fn checked_cdf(&self, x: f64) -> Result<f64>;
}

/// The `Continuous` trait extends the `Univariate`
/// trait and provides an interface for interacting with continuous
/// univariate statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Continuous` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution. The `CheckedContinuous`
/// trait provides a panic-safe interface for continuous distributions
pub trait Continuous : Univariate {
    /// Returns the mode for a given distribution. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Continuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.mode());
    /// ```
    fn mode(&self) -> f64;

    /// Returns the minimum value in the domain of a given distribution 
    /// representable by a double-precision float. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Continuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.min());
    /// ```
    fn min(&self) -> f64;

    /// Returns the maximum value in the domain of a given distribution 
    /// representable by a double-precision float. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Continuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.max());
    /// ```
    fn max(&self) -> f64;

    /// Returns the probability density function calculated at `x` for a given distribution. 
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Continuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.pdf(0.5));
    /// ```
    fn pdf(&self, x: f64) -> f64;

    /// Returns the log of the probability density function calculated at `x` for a given distribution. 
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Continuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.ln_pdf(0.5));
    /// ```
    fn ln_pdf(&self, x: f64) -> f64;
}

/// The `CheckedContinous` trait extends the `CheckedUnivariate`
/// trait and provides a checked interface for interacting with
/// continous univariate statistical distributions. This means 
/// implementors should return an `StatsError` instead of panicking on
/// an invalid state or input.
pub trait CheckedContinuous : CheckedUnivariate {
    /// Returns the mode for a given distribution or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedContinuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.checked_mode().unwrap());
    /// ```
    fn checked_mode(&self) -> Result<f64>;

    /// Returns the minimum value in the domain of a given distribution 
    /// representable by a double-precision float or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedContinuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.checked_min().unwrap());
    /// ```
    fn checked_min(&self) -> Result<f64>;

    /// Returns the maximum value in the domain of a given distribution 
    /// representable by a double-precision float or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedContinuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.checked_max().unwrap());
    /// ```
    fn checked_max(&self) -> Result<f64>;

    /// Returns the probability density function calculated at `x` 
    /// for a given distribution or a `StatsError` depending on the implementor. 
    /// This method should not panic but return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedContinuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.checked_pdf(0.5).unwrap());
    /// ```
    fn checked_pdf(&self, x: f64) -> Result<f64>;

    /// Returns the log of the probability density function calculated at `x` 
    /// for a given distribution or a `StatsError` depending on the implementor. 
    /// This method should not panic but return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedContinuous, Uniform};
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.checked_pdf(0.5).unwrap());
    /// ```
    fn checked_ln_pdf(&self, x: f64) -> Result<f64>;
}

/// The `Discrete` trait extends the `Univariate`
/// trait and provides an interface for interacting with discrete
/// univariate statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Discrete` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution. The `CheckedDiscrete` trait
/// provides a panic-safe interface for discrete distributions
pub trait Discrete : Univariate {
    /// Returns the mode for a given distribution. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Discrete, Binomial};
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_eq!(5, n.mode());
    /// ```
    fn mode(&self) -> i64;

    /// Returns the minimum value in the domain of a given distribution 
    /// representable by a 64-bit integer. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Discrete, Binomial};
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_eq!(0, n.min());
    /// ```
    fn min(&self) -> i64;

    /// Returns the maximum value in the domain of a given distribution 
    /// representable by a 64-bit integer. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Discrete, Binomial};
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_eq!(10, n.max());
    /// ```
    fn max(&self) -> i64;

    /// Returns the probability mass function calculated at `x` for a given distribution. 
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Discrete, Binomial};
    /// use statrs::prec;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert!(prec::almost_eq(n.pmf(5), 0.24609375, 1e-15));
    /// ```
    fn pmf(&self, x: i64) -> f64;

    /// Returns the log of the probability mass function calculated at `x` for a given distribution. 
    /// May panic depending on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{Discrete, Binomial};
    /// use statrs::prec;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert!(prec::almost_eq(n.ln_pmf(5), (0.24609375f64).ln(), 1e-15));
    /// ```
    fn ln_pmf(&self, x: i64) -> f64;
}

/// The `CheckedDiscrete` trait extends the `CheckedUnivariate` trait
/// and provides a checked interface for interacting with discrete
/// univariate statistical distributions. This means implementors
/// should return an `StatsError` instead of panicking on an invalid state
/// or input
pub trait CheckedDiscrete : CheckedUnivariate {
    /// Returns the mode for a given distribution or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedDiscrete, Binomial};
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_eq!(5, n.checked_mode().unwrap());
    /// ```
    fn checked_mode(&self) -> Result<i64>;
    
    /// Returns the minimum value in the domain of a given distribution 
    /// representable by a double-precision float or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedDiscrete, Binomial};
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_eq!(0, n.checked_min().unwrap());
    /// ```
    fn checked_min(&self) -> Result<i64>;

    /// Returns the maximum value in the domain of a given distribution 
    /// representable by a double-precision float or a `StatsError`
    /// depending on the implementor. This method should not panic but
    /// return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedDiscrete, Binomial};
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert_eq!(10, n.checked_max().unwrap());
    /// ```
    fn checked_max(&self) -> Result<i64>;

    /// Returns the probability mass function calculated at `x` 
    /// for a given distribution or a `StatsError` depending on the implementor. 
    /// This method should not panic but return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedDiscrete, Binomial};
    /// use statrs::prec;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert!(prec::almost_eq(n.checked_pmf(5).unwrap(), 0.24609375, 1e-15));
    /// ```
    fn checked_pmf(&self, x: i64) -> Result<f64>;

    /// Returns the log of the probability mass function calculated at `x` 
    /// for a given distribution or a `StatsError` depending on the implementor. 
    /// This method should not panic but return a `StatsError` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::{CheckedDiscrete, Binomial};
    /// use statrs::prec;
    ///
    /// let n = Binomial::new(0.5, 10).unwrap();
    /// assert!(prec::almost_eq(n.checked_ln_pmf(5).unwrap(), (0.24609375f64).ln(), 1e-15));
    /// ```
    fn checked_ln_pmf(&self, x: i64) -> Result<f64>;
}
