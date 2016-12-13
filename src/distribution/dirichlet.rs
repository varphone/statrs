use result::Result;
use error::StatsError;

/// Implements the [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Dirichlet {
    alpha: Vec<f64>,
}

impl Dirichlet {
    /// Constructs a new dirichlet distribution with the given
    /// concenctration parameters (alpha)
    ///
    /// # Errors
    ///
    /// Returns an error if any element `x` in alpha exist
    /// such that `x < = 0.0` or if the length of alpha is
    /// less than 2
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Dirichlet;
    ///
    /// let alpha_ok = [1.0, 2.0, 3.0];
    /// let mut result = Dirichlet::new(&alpha_ok);
    /// assert!(result.is_ok());
    ///
    /// let alpha_err = [0.0];
    /// result = Dirichlet::new(&alpha_err);
    /// assert!(result.is_err());
    /// ```
    pub fn new(alpha: &[f64]) -> Result<Dirichlet> {
        if alpha.len() < 2 {
            return Err(StatsError::BadParams);
        }
        for x in alpha {
            if *x <= 0.0 {
                return Err(StatsError::BadParams);
            }
        }
        Ok(Dirichlet { alpha: alpha.to_vec() })
    }

    /// Returns the concentration parameters of
    /// the dirichlet distribution as a slice
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Dirichlet;
    ///
    /// let n = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(n.alpha(), [1.0, 2.0, 3.0]);
    /// ```
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }
}
