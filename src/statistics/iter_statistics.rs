use std::f64;
use std::borrow::Borrow;

/// The `IterStatistics` trait provides the same host of statistical
/// utilities as the `Statistics` traited ported for use with iterators
/// which requires a mutable borrow
pub trait IterStatistics<T> {
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
    /// use statrs::statistics::IterStatistics;
    ///
    /// let x: Vec<f64> = vec![];
    /// assert!(x.iter().abs_min().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.iter().abs_min().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.iter().abs_min(), 0.0);
    /// ```
    fn abs_min(mut self) -> T;

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
    /// use statrs::statistics::IterStatistics;
    ///
    /// let x: Vec<f64> = vec![];
    /// assert!(x.iter().abs_max().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.iter().abs_max().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0, -8.0];
    /// assert_eq!(z.iter().abs_max(), 8.0);
    /// ```
    fn abs_max(mut self) -> T;

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
    /// use statrs::statistics::IterStatistics;
    ///
    /// # fn main() {
    /// let x: Vec<f64> = vec![];
    /// assert!(x.iter().geometric_mean().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.iter().geometric_mean().is_nan());
    ///
    /// let mut z = [0.0, 3.0, -2.0];
    /// assert!(z.iter().geometric_mean().is_nan());
    ///
    /// z = [0.0, 3.0, 2.0];
    /// assert_eq!(z.iter().geometric_mean(), 0.0);
    ///
    /// z = [1.0, 2.0, 3.0];
    /// // test value from online calculator, could be more accurate
    /// assert_almost_eq!(z.iter().geometric_mean(), 1.81712, 1e-5);
    /// # }
    /// ```
    fn geometric_mean(mut self) -> T;
}

impl<T> IterStatistics<f64> for T
    where T: Iterator,
          T::Item: Borrow<f64>
{
    fn abs_min(mut self) -> f64 {
        match self.next() {
            None => f64::NAN,
            Some(x) => {
                self.map(|x| x.borrow().abs())
                    .fold(x.borrow().abs(),
                          |acc, x| if x < acc || x.is_nan() { x } else { acc })
            }
        }
    }

    fn abs_max(mut self) -> f64 {
        match self.next() {
            None => f64::NAN,
            Some(x) => {
                self.map(|x| x.borrow().abs())
                    .fold(x.borrow().abs(),
                          |acc, x| if x > acc || x.is_nan() { x } else { acc })
            }
        }
    }

    fn geometric_mean(self) -> f64 {
        let mut count = 0.0;
        let mut sum = 0.0;
        for x in self {
            count += 1.0;
            sum += x.borrow().ln();
        }
        if count > 0.0 {
            (sum / count).exp()
        } else {
            f64::NAN
        }
    }
}
