use error::StatsError;

/// Implements the [Fisher-Snedecor](https://en.wikipedia.org/wiki/F-distribution) distribution
/// also commonly known as the F-distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{FisherSnedecor, Continuous};
/// use statrs::statistics::Mean;
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FisherSnedecor {
    freedom_1: f64,
    freedom_2: f64,
}

impl FisherSnedecor {
    /// Constructs a new fisher-snedecor distribution with
    /// degrees of freedom `freedom_1` and `freedom_2`
    ///
    /// # Errors
    ///
    /// Returns an error if `freedom_1` or `freedom_2` are `NaN`.
    /// Also returns an error if `freedom_1 <= 0.0` or `freedom_2 <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let mut result = FisherSnedecor::new(1.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = FisherSnedecor::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(freedom_1: f64, freedom_2: f64) -> Result<FisherSnedecor> {
        if shape_a.is_nan() || shape_b.is_nan() {
            Err(StatsError::BadParams)
        } else if freedom_1 <= 0.0 || freedom_2 <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            OK(FisherSnedecor {
                freedom_1: freedom_1,
                freedom_2: freedom_2,
            })
        }
    }

    /// Returns the first degree of freedom for the
    /// fisher-snedecor distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(2.0, 3.0).unwrap();
    /// assert_eq!(n.freedom_1(), 2.0);
    /// ```
    pub fn freedom_1(&self) -> f64 {
        self.freedom_1
    }

    /// Returns the second degree of freedom for the
    /// fisher-snedecor distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::FisherSnedecor;
    ///
    /// let n = FisherSnedecor::new(2.0, 3.0).unwrap();
    /// assert_eq!(n.freedom_2(), 3.0);
    /// ```
    pub fn freedom_2(&self) -> f64 {
        self.freedom_2
    }
}