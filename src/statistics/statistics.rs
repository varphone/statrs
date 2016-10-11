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

/// The `Statistics` trait provides a host of statistical utilities for analzying
/// data sets
pub trait Statistics {
    /// Returns the minimum absolute value in the data
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
    /// assert!(x.abs_max().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.abs_max().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0, -8.0];
    /// assert_eq!(z.abs_max(), 8.0);
    /// ```
    fn abs_max(&self) -> f64;

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

    /// Evaluates the population variance from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
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
    /// assert!(x.population_variance().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.population_variance().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.population_variance(), 38.0 / 9.0);
    /// ```
    fn population_variance(&self) -> f64;

    /// Evaluates the population standard deviation from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
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
    /// assert!(x.population_std_dev().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.population_std_dev().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.population_std_dev(), (38f64 / 9.0).sqrt());
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.covariance(&[]).is_nan());
    ///
    /// let y1 = [0.0, f64::NAN, 3.0, -2.0];
    /// let y2 = [-5.0, 4.0, 10.0, f64::NAN];
    /// assert!(y1.covariance(&y2).is_nan());
    ///
    /// let z1 = [0.0, 3.0, -2.0];
    /// let z2 = [-5.0, 4.0, 10.0];
    /// assert_eq!(z1.covariance(&z2), -5.5);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.population_covariance(&[]).is_nan());
    ///
    /// let y1 = [0.0, f64::NAN, 3.0, -2.0];
    /// let y2 = [-5.0, 4.0, 10.0, f64::NAN];
    /// assert!(y1.population_covariance(&y2).is_nan());
    ///
    /// let z1 = [0.0, 3.0, -2.0];
    /// let z2 = [-5.0, 4.0, 10.0];
    /// assert_eq!(z1.population_covariance(&z2), -11.0 / 3.0);
    /// ```
    fn population_covariance(&self, other: &Self) -> f64;

    /// Estimates the quadratic mean (Root Mean Square) of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or any entry is `f64::NAN`
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
    /// assert!(x.quadratic_mean().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.quadratic_mean().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// // test value from online calculator, could be more accurate
    /// assert_almost_eq!(z.quadratic_mean(), 2.08167, 1e-5);
    /// # }
    /// ```
    fn quadratic_mean(&self) -> f64;

    /// Returns the order statistic `(order 1..N)` from the data
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Order must be one-based (between `1` and `N` inclusive)
    /// Returns `f64::NAN` if order is outside the viable range or data is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.order_statistic(1).is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// assert!(y.order_statistic(0).is_nan());
    /// assert!(y.order_statistic(4).is_nan());
    /// assert_eq!(y.order_statistic(2), 0.0);
    /// ```
    fn order_statistic(&self, order: usize) -> f64;

    /// Estimates the tau-th quantile from the data. The tau-th quantile
    /// is the data value where the cumulative distribution function crosses tau.
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Tau must be between `0` and `1` inclusive.
    /// Returns `f64::NAN` if data is empty or tau is outside the inclusive range.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.quantile(0.5).is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// assert!(y.quantile(-1.0).is_nan());
    /// assert!(y.quantile(2.0).is_nan());
    /// assert_eq!(y.quantile(0.5), 0.0);
    /// ```
    fn quantile(&self, tau: f64) -> f64;

    /// Estimates the p-Percentile value from the data.
    ///
    /// # Remarks
    ///
    /// Use quantile for non-integer percentiles. `p` must be between `0` and `100` inclusive.
    /// Returns `f64::NAN` if data is empty or `p` is outside the inclusive range.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Statistics;
    ///
    /// let x = [];
    /// assert!(x.percentile(0).is_nan());
    ///
    /// let y = [1.0, 5.0, 3.0, 4.0, 10.0, 9.0, 6.0, 7.0, 8.0, 2.0];
    /// assert_eq!(y.percentile(0), 1.0);
    /// assert_eq!(y.percentile(50), 5.5);
    /// assert_eq!(y.percentile(100), 10.0);
    /// assert!(y.percentile(105).is_nan());
    /// ```
    fn percentile(&self, p: usize) -> f64;

    /// Estimates the first quartile value from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.lower_quartile().is_nan());
    ///
    /// let y = [2.0, 1.0, 3.0, 4.0];
    /// assert_almost_eq!(y.lower_quartile(), 1.416666666666666, 1e-15);
    /// # }
    /// ```
    fn lower_quartile(&self) -> f64;

    /// Estimates the third quartile value from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.lower_quartile().is_nan());
    ///
    /// let y = [2.0, 1.0, 3.0, 4.0];
    /// assert_almost_eq!(y.upper_quartile(), 3.5833333333333333, 1e-15);
    /// # }
    /// ```
    fn upper_quartile(&self) -> f64;

    /// Estimates the inter-quartile range from the data.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::statistics::Statistics;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.interquartile_range().is_nan());
    ///
    /// let y = [2.0, 1.0, 3.0, 4.0];
    /// assert_almost_eq!(y.interquartile_range(), 2.166666666666667, 1e-15);
    /// # }
    /// ```
    fn interquartile_range(&self) -> f64;

    /// Evaluates the rank of each entry of the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::{Statistics, RankTieBreaker};
    ///
    /// let x = [];
    /// assert_eq!(x.ranks(RankTieBreaker::Average).len(), 0);
    ///
    /// let y = [1.0, 3.0, 2.0, 2.0];
    /// assert_eq!((&y.clone()).ranks(RankTieBreaker::Average), [1.0, 4.0, 2.5, 2.5]);
    /// assert_eq!((&y.clone()).ranks(RankTieBreaker::Min), [1.0, 4.0, 2.0, 2.0]);
    /// ```
    fn ranks(&self, tie_breaker: RankTieBreaker) -> Vec<f64>;
}
