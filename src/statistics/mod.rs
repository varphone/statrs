//! Provides statistical computation utilities for data sets

mod slice_statistics;

/// Enumeration of possible tie-breaking strategies
/// when computing ranks
#[derive(Debug, Copy, Clone)]
pub enum RankTieBreaker {
    /// Replaces ties with their mean
    Average,
    /// Replace ties with their minimum
    Min,
    /// Replace ties with their maximum
    Max,
    /// Permutation with increasing values at each index of ties
    First,
}

/// The statistics trait provides a host of statistical utilities for analzying
/// data sets
pub trait Statistics {
    /// Returns the minimum value in the data
    ///
    /// # Rermarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.min().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.min().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.min(), -2.0);
    /// ```
    fn min(&self) -> f64;

    /// Returns the maximum value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.max().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.max().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.max(), 3.0);
    /// ```
    fn max(&self) -> f64;

    /// Returns the minimum absolute value in the data
    ///
    /// # Rermarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.abs_min().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.abs_min().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.abs_min(), 0.0);
    /// ```
    fn abs_min(&self) -> f64;

    /// Returns the maximum absolute value in the data
    ///
    /// # Rermarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.abs_max().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.abs_max().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0, -8.0];
    /// assert_eq!(z.abs_max(), 8.0);
    /// ```
    fn abs_max(&self) -> f64;

    /// Evaluates the sample mean, an estimate of the population
    /// mean.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.mean().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.mean().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_almost_eq!(z.mean(), 1.0 / 3.0, 1e-15);
    /// # }
    /// ```
    fn mean(&self) -> f64;

    /// Evaluates the geometric mean of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`.
    /// Returns `f64::NAN` if an entry is less than `0`. Returns `0`
    /// if no entry is less than `0` but there are entries equal to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.geometric_mean().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.geometric_mean().is_nan());
    ///
    /// let mut z = [0.0, 3.0, -2.0];
    /// assert!(z.geometric_mean().is_nan());
    ///
    /// z = [0.0, 3.0, 2.0];
    /// assert_eq!(z.geometric_mean(), 0.0);
    ///
    /// z = [1.0, 2.0, 3.0];
    /// // test value from online calculator, could be more accurate
    /// assert_almost_eq!(z.geometric_mean(), 1.81712, 1e-5);
    /// # }
    /// ```
    fn geometric_mean(&self) -> f64;

    /// Evaluates the harmonic mean of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`, or if any value
    /// in data is less than `0`. Returns `0` if there are no values less than `0` but
    /// there exists values equal to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.harmonic_mean().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.harmonic_mean().is_nan());
    ///
    /// let mut z = [0.0, 3.0, -2.0];
    /// assert!(z.harmonic_mean().is_nan());
    ///
    /// z = [0.0, 3.0, 2.0];
    /// assert_eq!(z.harmonic_mean(), 0.0);
    ///
    /// z = [1.0, 2.0, 3.0];
    /// // test value from online calculator, could be more accurate
    /// assert_almost_eq!(z.harmonic_mean(), 1.63636, 1e-5);
    /// # }
    /// ```
    fn harmonic_mean(&self) -> f64;

    /// Estimates the unbiased population variance from the provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.variance().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, 2.0];
    /// assert!(y.variance().is_nan());
    /// ```
    fn variance(&self) -> f64;

    /// Evaluates the population variance from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn population_variance(&self) -> f64;

    /// Estimates the unbiased population standard deviation from the provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is `f64::NAN`
    fn std_dev(&self) -> f64;

    /// Evaluates the population standard deviation from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn population_std_dev(&self) -> f64;

    /// Estimates the unbiased population covariance between the two provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is `f64::NAN`
    ///
    /// # Panics
    ///
    /// If the two sample containers do not contain the same number of elements
    fn covariance(&self, other: &Self) -> f64;

    /// Evaluates the population covariance between the two provider populations
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus be
    /// biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or any entry is `f64::NAN`
    ///
    /// # Panics
    ///
    /// If the two sample containers do not contain the same number of elements
    fn population_covariance(&self, other: &Self) -> f64;

    /// Estimates the quadratic mean (Root Mean Square) of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or any entry is `f64::NAN`
    fn quadratic_mean(&self) -> f64;

    /// Returns the order statistic `(order 1..N)` from the data
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Order must be one-based (between `1` and `N` inclusive)
    /// Returns `f64::NAN` if order is outside the viable range or data is empty.
    fn order_statistic(&mut self, order: usize) -> f64;

    /// Returns the median value from the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    fn median(&mut self) -> f64;

    /// Estimates the tau-th quantile from the data. The tau-th quantile
    /// is the data value where the cumulative distribution function crosses tau.
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Tau must be between `0` and `1` inclusive.
    /// Returns `f64::NAN` if data is empty or tau is outside the inclusive range.
    fn quantile(&mut self, tau: f64) -> f64;

    /// Estimates the p-Percentile value from the data.
    ///
    /// # Remarks
    ///
    /// Use quantile for non-integer percentiles. `p` must be between `0` and `100` inclusive.
    /// Returns `f64::NAN` if data is empty or `p` is outside the inclusive range.
    fn percentile(&mut self, p: usize) -> f64;

    /// Estimates the first quartile value from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    fn lower_quartile(&mut self) -> f64;

    /// Estimates the third quartile value from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    fn upper_quartile(&mut self) -> f64;

    /// Estimates the inter-quartile range from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    fn interquartile_range(&mut self) -> f64;

    /// Evaluates the rank of each entry of the data.
    fn ranks(&mut self, tie_breaker: RankTieBreaker) -> Vec<f64>;
}
