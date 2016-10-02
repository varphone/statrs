//! This crate aims to be a functional
//! port of the Math.NET Numerics Distribution package and in doing so providing the Rust numerical
//! computing community with a robust, well-tested statistical distribution package. This crate
//! also ports over some of the special statistical functions from Math.NET in so far as they are
//! used in the computation of distribution values. This crate depends on the `rand` crate to provide
//! RNG.
//!
//! # Example
//! The following example samples from a standard normal distribution
//!
//! ```
//! # extern crate rand;
//! # extern crate statrs;

//! use rand::StdRng;
//! use statrs::distribution::{Distribution, Normal};
//!
//! # fn main() {
//! let mut r = rand::StdRng::new().unwrap();
//! let n = Normal::new(0.0, 1.0).unwrap();
//! for _ in 0..10 {
//!     print!("{}", n.sample::<StdRng>(&mut r));
//! }
//! # }
//! ```

#![crate_type = "lib"]
#![crate_name = "statrs"]

extern crate rand;

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $prec:expr) => (
        if !$crate::prec::almost_eq($a, $b, $prec) {
            panic!(format!("assertion failed: `abs(left - right) < {:e}`, (left: `{}`, right: `{}`)", $prec, $a, $b));
        }
    );
}

pub mod distribution;
pub mod euclid;
pub mod function;
pub mod generate;
pub mod consts;
pub mod prec;
pub mod statistics;

mod result;
mod error;

#[cfg(test)]
mod testing;

pub use result::Result;
pub use error::StatsError;

/// The `Mean` trait specifies than an object has a closed form
/// solution for its mean(s)
pub trait Mean<T> {
    /// Returns the mean. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::Mean;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.mean());
    /// ```
    fn mean(&self) -> T;
}

/// The `Variance` trait specifies that an object has a closed form solution for
/// its variance(s). Requires `Mean` since a closed form solution to
/// variance by definition requires a closed form mean.
pub trait Variance<T>: Mean<T> {
    /// Returns the variance. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::Variance;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0 / 12.0, n.variance());
    /// ```
    fn variance(&self) -> T;

    /// Returns the standard deviation. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::Variance;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!((1f64 / 12f64).sqrt(), n.std_dev());
    /// ```
    fn std_dev(&self) -> T;
}

/// The `Median` trait specifies than an object has a closed form solution
/// for its median
pub trait Median<T> {
    /// Returns the median. May panic depending
    /// on the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::Median;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.median());
    /// ```
    fn median(&self) -> T;
}

/// The `Mode` trait specififies than an object has a closed form solution
/// for its mode(s)
pub trait Mode<T> {
    /// Returns the mode. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::Mode;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.5, n.mode());
    /// ```
    fn mode(&self) -> T;
}

/// The `Min` trait specifies than an object has a minimum value
pub trait Min<T> {
    /// Returns the minimum value in the domain of a given distribution
    /// representable by a double-precision float. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::Min;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(0.0, n.min());
    /// ```
    fn min(&self) -> T;
}

/// The `Max` trait specifies that an object has a maximum value
pub trait Max<T> {
    /// Returns the maximum value in the domain of a given distribution
    /// representable by a double-precision float. May panic depending on
    /// the implementor.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::Max;
    /// use statrs::distribution::Uniform;
    ///
    /// let n = Uniform::new(0.0, 1.0).unwrap();
    /// assert_eq!(1.0, n.max());
    /// ```
    fn max(&self) -> T;
}
