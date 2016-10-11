use std::f64;

/// The `IterStatistics` trait provides the same host of statistical
/// utilities as the `Statistics` traited ported for use with iterators 
pub trait IterStatistics : Iterator<Item=f64>
{
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
    /// assert!(x.into_iter().abs_min().is_nan());
    ///
    /// let y: Vec<f64>  = vec![0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.into_iter().abs_min().is_nan());
    ///
    /// let z: Vec<f64>  = vec![0.0, 3.0, -2.0];
    /// assert_eq!(z.into_iter().abs_min(), 0.0);
    /// ```
    fn abs_min(&mut self) -> f64 {
        let mut min = f64::INFINITY;
        let mut any = false;
        loop {
            match self.next() {
                Some(x) => {
                    let abs = x.abs();
                    if abs < min || abs.is_nan() {
                        min = abs;
                    }
                    any = true;
                }
                None => break
            }
        }
        if any { min } else { f64::NAN }
    }
}

impl IterStatistics for Iterator<Item=f64> { }