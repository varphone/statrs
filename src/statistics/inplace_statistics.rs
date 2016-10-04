use super::RankTieBreaker;

/// The `InplaceStatistics` trait provides in place variations
/// of statistical utilities with side-effects allowing you to opt
/// into a little more efficiency at the cost of a mutable borrow
pub trait InplaceStatistics {
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
    /// use statrs::statistics::InplaceStatistics;
    ///
    /// let mut x = [];
    /// assert!(x.order_statistic_inplace(1).is_nan());
    ///
    /// let mut y = [0.0, 3.0, -2.0];
    /// assert!(y.order_statistic_inplace(0).is_nan());
    /// assert!(y.order_statistic_inplace(4).is_nan());
    /// assert_eq!(y.order_statistic_inplace(2), 0.0);
    /// assert!(y != [0.0, 3.0, -2.0]);
    /// ```
    fn order_statistic_inplace(&mut self, order: usize) -> f64;

    /// Returns the median value from the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::InplaceStatistics;
    ///
    /// let mut x = [];
    /// assert!(x.median_inplace().is_nan());
    ///
    /// let mut y = [0.0, 3.0, -2.0];
    /// assert_eq!(y.median_inplace(), 0.0);
    /// assert!(y != [0.0, 3.0, -2.0]);
    fn median_inplace(&mut self) -> f64;

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
    /// use statrs::statistics::InplaceStatistics;
    ///
    /// let mut x = [];
    /// assert!(x.quantile_inplace(0.5).is_nan());
    ///
    /// let mut y = [0.0, 3.0, -2.0];
    /// assert!(y.quantile_inplace(-1.0).is_nan());
    /// assert!(y.quantile_inplace(2.0).is_nan());
    /// assert_eq!(y.quantile_inplace(0.5), 0.0);
    /// assert!(y != [0.0, 3.0, -2.0]);
    /// ```
    fn quantile_inplace(&mut self, tau: f64) -> f64;

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
    /// use statrs::statistics::InplaceStatistics;
    ///
    /// let mut x = [];
    /// assert!(x.percentile_inplace(0).is_nan());
    ///
    /// let mut y = [1.0, 5.0, 3.0, 4.0, 10.0, 9.0, 6.0, 7.0, 8.0, 2.0];
    /// assert_eq!(y.percentile_inplace(0), 1.0);
    /// assert_eq!(y.percentile_inplace(50), 5.5);
    /// assert_eq!(y.percentile_inplace(100), 10.0);
    /// assert!(y.percentile_inplace(105).is_nan());
    /// assert!(y != [1.0, 5.0, 3.0, 4.0, 10.0, 9.0, 6.0, 7.0, 8.0, 2.0]);
    /// ```
    fn percentile_inplace(&mut self, p: usize) -> f64;

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
    /// use statrs::statistics::InplaceStatistics;
    ///
    /// # fn main() {
    /// let mut x = [];
    /// assert!(x.lower_quartile_inplace().is_nan());
    ///
    /// let mut y = [2.0, 1.0, 3.0, 4.0];
    /// assert_almost_eq!(y.lower_quartile_inplace(), 1.416666666666666, 1e-15);
    /// assert!(y != [2.0, 1.0, 3.0, 4.0]);
    /// # }
    /// ```
    fn lower_quartile_inplace(&mut self) -> f64;

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
    /// use statrs::statistics::InplaceStatistics;
    ///
    /// # fn main() {
    /// let mut x = [];
    /// assert!(x.upper_quartile_inplace().is_nan());
    ///
    /// let mut y = [2.0, 1.0, 3.0, 4.0];
    /// assert_almost_eq!(y.upper_quartile_inplace(), 3.5833333333333333, 1e-15);
    /// assert!(y != [2.0, 1.0, 3.0, 4.0]);
    /// # }
    /// ```
    fn upper_quartile_inplace(&mut self) -> f64;

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
    /// use statrs::statistics::InplaceStatistics;
    ///
    /// # fn main() {
    /// let mut x = [];
    /// assert!(x.interquartile_range_inplace().is_nan());
    ///
    /// let mut y = [2.0, 1.0, 3.0, 4.0];
    /// assert_almost_eq!(y.interquartile_range_inplace(), 2.166666666666667, 1e-15);
    /// assert!(y != [2.0, 1.0, 3.0, 4.0]);
    /// # }
    /// ```
    fn interquartile_range_inplace(&mut self) -> f64;

    /// Evaluates the rank of each entry of the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::{InplaceStatistics, RankTieBreaker};
    ///
    /// let mut x = [];
    /// assert_eq!(x.ranks_inplace(RankTieBreaker::Average).len(), 0);
    ///
    /// let y = [1.0, 3.0, 2.0, 2.0];
    /// assert_eq!((&mut y.clone()).ranks_inplace(RankTieBreaker::Average), [1.0, 4.0, 2.5, 2.5]);
    /// assert_eq!((&mut y.clone()).ranks_inplace(RankTieBreaker::Min), [1.0, 4.0, 2.0, 2.0]);
    /// ```
    fn ranks_inplace(&mut self, tie_breaker: RankTieBreaker) -> Vec<f64>;
}
