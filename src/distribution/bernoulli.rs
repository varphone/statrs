use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use result::Result;
use super::{Binomial, Distribution, Univariate, Discrete};

/// Implements the [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)
/// distribution which is a special case of the [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution where `n = 1` (referenced [Here](./struct.Binomial.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Bernoulli, Univariate, Discrete};
///
/// let n = Bernoulli::new(0.5).unwrap(); 
/// assert_eq!(n.mean(), 0.5);
/// assert_eq!(n.pmf(0), 0.5);
/// assert_eq!(n.pmf(1), 0.5);
/// assert_eq!(n.pmf(2), 0.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Bernoulli {
    b: Binomial,
}

impl Bernoulli {
    /// Constructs a new Bernoulli distribution with
    /// the given `p` probability of success.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is `NaN`, less than `0.0`
    /// or greater than `1.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    ///
    /// let mut result = Bernoulli::new(0.5);
    /// assert!(result.is_ok());
    ///
    /// result = Bernoulli::new(-0.5);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: f64) -> Result<Bernoulli> {
        Binomial::new(p, 1).map(|b| Bernoulli { b: b })
    }

    /// Returns the probability of success `p` of the
    /// Bernoulli distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    ///
    /// let n = Bernoulli::new(0.5).unwrap();
    /// assert_eq!(n.p(), 0.5);
    /// ```
    pub fn p(&self) -> f64 {
        self.b.p()
    }

    /// Returns the number of trials `n` of the
    /// Bernoulli distribution. Will always be `1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Bernoulli;
    ///
    /// let n = Bernoulli::new(0.5).unwrap();
    /// assert_eq!(n.n(), 1);
    /// ```
    pub fn n(&self) -> i64 {
        1
    }
}

impl Sample<f64> for Bernoulli {
    /// Generate a random sample from a Bernoulli
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Bernoulli {
    /// Generate a random independent sample from a Bernoulli
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution for Bernoulli {
    /// Generate a random sample from the
    /// Bernoulli distribution using `r` as the source
    /// of randomness where the generated
    /// values are `1` with probability `p` and `0`
    /// with probability `1-p`.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Bernoulli, Distribution};
    /// 
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Bernoulli::new(0.5).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        self.b.sample(r)
    }
}

impl Univariate for Bernoulli {
    /// Returns the mean of the Bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// p
    /// ```
    fn mean(&self) -> f64 {
        self.b.mean()
    }

    /// Returns the variance of the Bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// p * (1 - p)
    /// ```
    fn variance(&self) -> f64 {
        self.b.variance()
    }

    /// Returns the standard deviation of the
    /// Bernoulli distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(p * (1 - p))
    /// ```
    fn std_dev(&self) -> f64 {
        self.b.std_dev()
    }

    /// Returns the entropy of the Bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// q = (1 - p)
    /// -q * ln(q) - p * ln(p)
    /// ```
    fn entropy(&self) -> f64 {
        self.b.entropy()
    }

    /// Returns the skewness of the Bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// q = (1 - p)
    /// (1 - 2p) / sqrt(p * q)
    /// ```
    fn skewness(&self) -> f64 {
        self.b.skewness()
    }

    /// Returns the median of the Bernoulli
    /// distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if p < 0.5 { 0 }
    /// else if p > 0.5 { 1 }
    /// else { 0.5 }
    /// ```
    fn median(&self) -> f64 {
        self.b.median()
    }

    /// Calculates the cumulative distribution
    /// function for the Bernoulli distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x < 0.0` and `1.0` if `x >= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < 0 { 0 }
    /// else if x >= 1 { 1 }
    /// else { 1 - p }
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        self.b.cdf(x)
    }
}

impl Discrete for Bernoulli {
    /// Returns the mode of the Bernoulli distribution
    ///
    /// # Formula
    /// 
    /// ```ignore
    /// if p < 0.5 { 0 }
    /// else { 1 }
    /// ```
    fn mode(&self) -> i64 {
        self.b.mode()
    }

    /// Returns the minimum value in the domain of the
    /// Bernoulli distribution representable by a 64-
    /// bit integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> i64 {
        0
    }

    /// Returns the maximum value in the domain of the
    /// Bernoulli distribution representable by a 64-
    /// bit integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1
    /// ```
    fn max(&self) -> i64 {
        1
    }

    /// Calculates the probability mass function for the
    /// Bernoulli distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x < 0 || x > 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < 0 || x > 1 { 0.0 }
    /// else if x == 0 { 1 - p }
    /// else { p }
    /// ```
    fn pmf(&self, x: i64) -> f64 {
        self.b.pmf(x)
    }

    /// Calculates the log probability mass function for the
    /// Bernoulli distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NEG_INFINITY` if `x < 0 || x > 1`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < 0 || x > 1 { -INF }
    /// else if x == 0 { ln(1 - p) }
    /// else { ln(p) }
    /// ```
    fn ln_pmf(&self, x: i64) -> f64 {
        self.b.ln_pmf(x)
    }
}
