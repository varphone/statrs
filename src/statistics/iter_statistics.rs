use std::f64;

/// The `IterStatistics` trait provides the same host of statistical
/// utilities as the `Statistics` traited ported for use with iterators
/// which requires a mutable borrow
pub trait IterStatistics<I, T> : Iterator<Item=I> {
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
    /// let y = vec![0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.iter().abs_min().is_nan());
    ///
    /// let z = vec![0.0, 3.0, -2.0];
    /// assert_eq!(z.iter().abs_min(), 0.0);
    /// ```
    fn abs_min(&mut self) -> T;

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
    /// assert!(x.into_iter().abs_max().is_nan());
    ///
    /// let y = vec![0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.into_iter().abs_max().is_nan());
    ///
    /// let z = vec![0.0, 3.0, -2.0, -8.0];
    /// assert_eq!(z.into_iter().abs_max(), 8.0);
    /// ```
    fn abs_max(&mut self) -> T;
}

impl<'a, T: Iterator<Item=&'a f64>> IterStatistics<&'a f64, f64> for T 
{
    fn abs_min(&mut self) -> f64 {
        match self.next() {
            None => f64::NAN,
            Some(x) => {
                self.map(|x| x.abs())
                    .fold(x.abs(),
                          |acc, x| if x < acc || x.is_nan() { x } else { acc })
            }
        }
    }

    fn abs_max(&mut self) -> f64 {
        match self.next() {
            None => f64::NAN,
            Some(x) => {
                self.map(|x| x.abs())
                    .fold(x.abs(),
                          |acc, x| if x > acc || x.is_nan() { x } else { acc })
            }
        }
    }
}

impl<'a, T: Iterator<Item=f64>> IterStatistics<f64, f64> for T {
    fn abs_min(&mut self) -> f64 {
        match self.next() {
            None => f64::NAN,
            Some(x) => {
                self.map(|x| x.abs())
                    .fold(x.abs(),
                          |acc, x| if x < acc || x.is_nan() { x } else { acc })
            }
        }
    }

    fn abs_max(&mut self) -> f64 {
        match self.next() {
            None => f64::NAN,
            Some(x) => {
                self.map(|x| x.abs())
                    .fold(x.abs(),
                          |acc, x| if x > acc || x.is_nan() { x } else { acc })
            }
        }
    }
}
