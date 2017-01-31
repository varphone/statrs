use {Result, StatsError};

/// Implements the [Hypergeometric](http://en.wikipedia.org/wiki/Hypergeometric_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draw: u64,
}

impl Hypergeometric {
    /// Constructs a new hypergeometric distribution
    /// with a population (N) of `population`, number
    /// of successes (K) of `successes`, and number of draws
    /// (n) of `draws`
    ///
    /// # Errors
    ///
    /// If `successes > population` or `draws > population`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let mut result = Hypergeometric::new(2, 2, 2);
    /// assert!(result.is_ok());
    ///
    /// result = Hypergeometric::new(2, 3, 2);
    /// assert!(result.is_err());
    /// ```
    pub fn new(population: u64, successes: u64, draw: u64) -> Result<Hypergeometric> {
        Err(StatsError::BadParams)
    }
}