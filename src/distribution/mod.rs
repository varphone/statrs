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
/// use rand::StdRng;
///
/// struct Foo;
///
/// impl Distribution for Foo {
///     fn sample<R: Rng>(&self, r: &mut R) -> f64 {
///         r.next_f64()
///     }
/// }
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
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0, 1).unwrap();
    /// assert_eq!(0.5, n.mean());
    /// ```
    fn mean(&self) -> f64;
    
    /// Returns the variance for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0, 1).unwrap();
    /// assert_eq!(1.0 / 12.0, n.variance());
    /// ```
    fn variance(&self) -> f64;
    
    /// Returns the standard deviation for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Uniform;
    /// 
    /// let n = Uniform::new(0, 1).unwrap();
    /// assert_eq!((1f64 / 12f64).sqrt(), n.std_dev());
    /// ```
    fn std_dev(&self) -> f64;
    
    /// Returns the entropy for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0, 1).unwrap();
    /// assert_eq!(0.0, n.entropy());
    /// ```
    fn entropy(&self) -> f64;
    
    /// Returns the skewness for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0, 1).unwrap();
    /// assert_eq!(0.0, n.skewness());
    /// ```
    fn skewness(&self) -> f64;
    
    /// Returns the median for a given distribution. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0, 1).unwrap();
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
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0, 1).unwrap();
    /// assert_eq!(0.5, n.cdf(0.5));
    /// ```
    fn cdf(&self, x: f64) -> f64;
}

/// The `CheckedUnivariate` trait extends the `Distribution` trait and
/// provides a checked interface for interacting with univariate statistical
/// distributions. This means implementors should return an `Err` instead of
/// panicking on an invalid state or input.
pub trait CheckedUnivariate : Distribution {
    fn mean(&self) -> Result<f64>;
    fn variance(&self) -> Result<f64>;
    fn std_dev(&self) -> Result<f64>;
    fn entropy(&self) -> Result<f64>;
    fn skewness(&self) -> Result<f64>;
    fn median(&self) -> Result<f64>;
    fn cdf(&self) -> Result<f64>;
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
    fn mode(&self) -> f64;
    fn min(&self) -> f64;
    fn max(&self) -> f64;
    fn pdf(&self, x: f64) -> f64;
    fn ln_pdf(&self, x: f64) -> f64;
}

/// The `CheckedContinous` trait extends the `CheckedUnivariate`
/// trait and provides a checked interface for interacting with
/// continous univariate statistical distributions. This means 
/// implementors should return an `Err` instead of panicking on
/// an invalid state or input.
pub trait CheckedContinuous : CheckedUnivariate {
    fn mode(&self) -> Result<f64>;
    fn min(&self) -> Result<f64>;
    fn max(&self) -> Result<f64>;
    fn pdf(&self, x: f64) -> Result<f64>;
    fn ln_pdf(&self, x: f64) -> Result<f64>;
}

/// The `Discrete` trait extends the `Univariate`
/// trait and provides an interface for interacting with discrete
/// univariate statistical distributions
///
/// # Remakrs
///
/// All methods provided by the `Discrete` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution. The `CheckedDiscrete` trait
/// provides a panic-safe interface for discrete distributions
pub trait Discrete : Univariate {
    fn mode(&self) -> i64;
    fn min(&self) -> i64;
    fn max(&self) -> i64;
    fn pmf(&self, x: i64) -> f64;
    fn ln_pmf(&self, x: i64) -> f64;
}

/// The `CheckedDiscrete` trait extends the `CheckedUnivariate` trait
/// and provides a checked interface for interacting with discrete
/// univariate statistical distributions. This means implementors
/// should return an `Err` instead of panicking on an invalid state
/// or input
pub trait CheckedDiscrete : CheckedUnivariate {
    fn mode(&self) -> Result<i64>;
    fn min(&self) -> Result<i64>;
    fn max(&self) -> Result<i64>;
    fn pmf(&self, x: i64) -> Result<f64>;
    fn ln_pmf(&self, x: i64) -> Result<f64>;
}