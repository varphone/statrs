use crate::distribution::{Discrete, DiscreteCDF};
use crate::function::factorial;
use crate::statistics::*;
use std::cmp;
use std::f64;

/// Implements the
/// [Hypergeometric](http://en.wikipedia.org/wiki/Hypergeometric_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Hypergeometric, Discrete};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = Hypergeometric::new(500, 50, 100).unwrap();
/// assert_eq!(n.mean().unwrap(), 10.);
/// assert!(prec::almost_eq(n.pmf(10), 0.14736784, 1e-8));
/// assert!(prec::almost_eq(n.pmf(25), 3.537e-7, 1e-10));
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draws: u64,
}

/// Represents the errors that can occur when creating a [`Hypergeometric`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum HypergeometricError {
    /// The number of successes is greater than the population.
    TooManySuccesses,

    /// The number of draws is greater than the population.
    TooManyDraws,
}

impl std::fmt::Display for HypergeometricError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HypergeometricError::TooManySuccesses => write!(f, "successes > population"),
            HypergeometricError::TooManyDraws => write!(f, "draws > population"),
        }
    }
}

impl std::error::Error for HypergeometricError {}

impl Hypergeometric {
    /// Constructs a new hypergeometric distribution
    /// with a population (N) of `population`, number
    /// of successes (K) of `successes`, and number of draws
    /// (n) of `draws`.
    ///
    /// # Errors
    ///
    /// If `successes > population` or `draws > population`.
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
    pub fn new(
        population: u64,
        successes: u64,
        draws: u64,
    ) -> Result<Hypergeometric, HypergeometricError> {
        if successes > population {
            return Err(HypergeometricError::TooManySuccesses);
        }

        if draws > population {
            return Err(HypergeometricError::TooManyDraws);
        }

        Ok(Hypergeometric {
            population,
            successes,
            draws,
        })
    }

    /// Returns the population size of the hypergeometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let n = Hypergeometric::new(10, 5, 3).unwrap();
    /// assert_eq!(n.population(), 10);
    /// ```
    pub fn population(&self) -> u64 {
        self.population
    }

    /// Returns the number of observed successes of the hypergeometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let n = Hypergeometric::new(10, 5, 3).unwrap();
    /// assert_eq!(n.successes(), 5);
    /// ```
    pub fn successes(&self) -> u64 {
        self.successes
    }

    /// Returns the number of draws of the hypergeometric
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Hypergeometric;
    ///
    /// let n = Hypergeometric::new(10, 5, 3).unwrap();
    /// assert_eq!(n.draws(), 3);
    /// ```
    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// Returns population, successes, and draws in that order
    /// as a tuple of doubles
    fn values_f64(&self) -> (f64, f64, f64) {
        (
            self.population as f64,
            self.successes as f64,
            self.draws as f64,
        )
    }
}

impl std::fmt::Display for Hypergeometric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hypergeometric({},{},{})",
            self.population, self.successes, self.draws
        )
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<u64> for Hypergeometric {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let mut population = self.population as f64;
        let mut successes = self.successes as f64;
        let mut draws = self.draws;
        let mut x = 0;
        loop {
            let p = successes / population;
            let next: f64 = rng.random();
            if next < p {
                x += 1;
                successes -= 1.0;
            }
            population -= 1.0;
            draws -= 1;
            if draws == 0 {
                break;
            }
        }
        x
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distr::Distribution<f64> for Hypergeometric {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.sample::<u64, _>(self) as f64
    }
}

impl DiscreteCDF<u64, f64> for Hypergeometric {
    /// Calculates the cumulative distribution function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 - ((n choose x+1) * (N-n choose K-x-1)) / (N choose K) * 3_F_2(1,
    /// x+1-K, x+1-n; k+2, N+x+2-K-n; 1)
    /// ```
    ///
    /// where `N` is population, `K` is successes, `n` is draws,
    /// and `p_F_q` is the
    /// [generalized hypergeometric function](https://en.wikipedia.org/wiki/Generalized_hypergeometric_function)
    ///
    /// Calculated as a discrete integral over the probability mass
    /// function evaluated from 0..x+1
    fn cdf(&self, x: u64) -> f64 {
        if x < self.min() {
            0.0
        } else if x >= self.max() {
            1.0
        } else {
            let k = x;
            let ln_denom = factorial::ln_binomial(self.population, self.draws);
            (0..k + 1).fold(0.0, |acc, i| {
                acc + (factorial::ln_binomial(self.successes, i)
                    + factorial::ln_binomial(self.population - self.successes, self.draws - i)
                    - ln_denom)
                    .exp()
            })
        }
    }

    /// Calculates the survival function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 - ((n choose x+1) * (N-n choose K-x-1)) / (N choose K) * 3_F_2(1,
    /// x+1-K, x+1-n; x+2, N+x+2-K-n; 1)
    /// ```
    ///
    /// where `N` is population, `K` is successes, `n` is draws,
    /// and `p_F_q` is the
    /// [generalized hypergeometric function](https://en.wikipedia.org/wiki/Generalized_hypergeometric_function)
    ///
    /// Calculated as a discrete integral over the probability mass
    /// function evaluated from (x+1)..max
    fn sf(&self, x: u64) -> f64 {
        if x < self.min() {
            1.0
        } else if x >= self.max() {
            0.0
        } else {
            let k = x;
            let ln_denom = factorial::ln_binomial(self.population, self.draws);
            (k + 1..=self.max()).fold(0.0, |acc, i| {
                acc + (factorial::ln_binomial(self.successes, i)
                    + factorial::ln_binomial(self.population - self.successes, self.draws - i)
                    - ln_denom)
                    .exp()
            })
        }
    }
}

impl Min<u64> for Hypergeometric {
    /// Returns the minimum value in the domain of the
    /// hypergeometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```text
    /// max(0, n + K - N)
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn min(&self) -> u64 {
        (self.draws + self.successes).saturating_sub(self.population)
    }
}

impl Max<u64> for Hypergeometric {
    /// Returns the maximum value in the domain of the
    /// hypergeometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```text
    /// min(K, n)
    /// ```
    ///
    /// where `K` is successes and `n` is draws
    fn max(&self) -> u64 {
        cmp::min(self.successes, self.draws)
    }
}

impl Distribution<f64> for Hypergeometric {
    /// Returns the mean of the hypergeometric distribution
    ///
    /// # None
    ///
    /// If `N` is `0`
    ///
    /// # Formula
    ///
    /// ```text
    /// K * n / N
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn mean(&self) -> Option<f64> {
        if self.population == 0 {
            None
        } else {
            Some(self.successes as f64 * self.draws as f64 / self.population as f64)
        }
    }

    /// Returns the variance of the hypergeometric distribution
    ///
    /// # None
    ///
    /// If `N <= 1`
    ///
    /// # Formula
    ///
    /// ```text
    /// n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn variance(&self) -> Option<f64> {
        if self.population <= 1 {
            None
        } else {
            let (population, successes, draws) = self.values_f64();
            let val = draws * successes * (population - draws) * (population - successes)
                / (population * population * (population - 1.0));
            Some(val)
        }
    }

    /// Returns the skewness of the hypergeometric distribution
    ///
    /// # None
    ///
    /// If `N <= 2`
    ///
    /// # Formula
    ///
    /// ```text
    /// ((N - 2K) * (N - 1)^(1 / 2) * (N - 2n)) / ([n * K * (N - K) * (N -
    /// n)]^(1 / 2) * (N - 2))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn skewness(&self) -> Option<f64> {
        if self.population <= 2 {
            None
        } else {
            let (population, successes, draws) = self.values_f64();
            let val = (population - 1.0).sqrt()
                * (population - 2.0 * draws)
                * (population - 2.0 * successes)
                / ((draws * successes * (population - successes) * (population - draws)).sqrt()
                    * (population - 2.0));
            Some(val)
        }
    }
}

impl Mode<Option<u64>> for Hypergeometric {
    /// Returns the mode of the hypergeometric distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// floor((n + 1) * (k + 1) / (N + 2))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn mode(&self) -> Option<u64> {
        Some(((self.draws + 1) * (self.successes + 1)) / (self.population + 2))
    }
}

impl Discrete<u64, f64> for Hypergeometric {
    /// Calculates the probability mass function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (K choose x) * (N-K choose n-x) / (N choose n)
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn pmf(&self, x: u64) -> f64 {
        if x > self.draws {
            0.0
        } else {
            factorial::binomial(self.successes, x)
                * factorial::binomial(self.population - self.successes, self.draws - x)
                / factorial::binomial(self.population, self.draws)
        }
    }

    /// Calculates the log probability mass function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((K choose x) * (N-K choose n-x) / (N choose n))
    /// ```
    ///
    /// where `N` is population, `K` is successes, and `n` is draws
    fn ln_pmf(&self, x: u64) -> f64 {
        factorial::ln_binomial(self.successes, x)
            + factorial::ln_binomial(self.population - self.successes, self.draws - x)
            - factorial::ln_binomial(self.population, self.draws)
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(population: u64, successes: u64, draws: u64; Hypergeometric; HypergeometricError);

    #[test]
    fn test_create() {
        create_ok(0, 0, 0);
        create_ok(1, 1, 1,);
        create_ok(2, 1, 1);
        create_ok(2, 2, 2);
        create_ok(10, 1, 1);
        create_ok(10, 5, 3);
    }

    #[test]
    fn test_bad_create() {
        test_create_err(2, 3, 2, HypergeometricError::TooManySuccesses);
        test_create_err(10, 5, 20, HypergeometricError::TooManyDraws);
        create_err(0, 1, 1);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Hypergeometric| x.mean().unwrap();
        test_exact(1, 1, 1, 1.0, mean);
        test_exact(2, 1, 1, 0.5, mean);
        test_exact(2, 2, 2, 2.0, mean);
        test_exact(10, 1, 1, 0.1, mean);
        test_exact(10, 5, 3, 15.0 / 10.0, mean);
    }

    #[test]
    fn test_mean_with_population_0() {
        test_none(0, 0, 0, |dist| dist.mean());
    }

    #[test]
    fn test_variance() {
        let variance = |x: Hypergeometric| x.variance().unwrap();
        test_exact(2, 1, 1, 0.25, variance);
        test_exact(2, 2, 2, 0.0, variance);
        test_exact(10, 1, 1, 81.0 / 900.0, variance);
        test_exact(10, 5, 3, 525.0 / 900.0, variance);
    }

    #[test]
    fn test_variance_with_pop_lte_1() {
        test_none(1, 1, 1, |dist| dist.variance());
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Hypergeometric| x.skewness().unwrap();
        test_exact(10, 1, 1, 8.0 / 3.0, skewness);
        test_exact(10, 5, 3, 0.0, skewness);
    }

    #[test]
    fn test_skewness_with_pop_lte_2() {
        test_none(2, 2, 2, |dist| dist.skewness());
    }

    #[test]
    fn test_mode() {
        let mode = |x: Hypergeometric| x.mode().unwrap();
        test_exact(0, 0, 0, 0, mode);
        test_exact(1, 1, 1, 1, mode);
        test_exact(2, 1, 1, 1, mode);
        test_exact(2, 2, 2, 2, mode);
        test_exact(10, 1, 1, 0, mode);
        test_exact(10, 5, 3, 2, mode);
    }

    #[test]
    fn test_min() {
        let min = |x: Hypergeometric| x.min();
        test_exact(0, 0, 0, 0, min);
        test_exact(1, 1, 1, 1, min);
        test_exact(2, 1, 1, 0, min);
        test_exact(2, 2, 2, 2, min);
        test_exact(10, 1, 1, 0, min);
        test_exact(10, 5, 3, 0, min);
    }

    #[test]
    fn test_max() {
        let max = |x: Hypergeometric| x.max();
        test_exact(0, 0, 0, 0, max);
        test_exact(1, 1, 1, 1, max);
        test_exact(2, 1, 1, 1, max);
        test_exact(2, 2, 2, 2, max);
        test_exact(10, 1, 1, 1, max);
        test_exact(10, 5, 3, 3, max);
    }

    #[test]
    fn test_pmf() {
        let pmf = |arg: u64| move |x: Hypergeometric| x.pmf(arg);
        test_exact(0, 0, 0, 1.0, pmf(0));
        test_exact(1, 1, 1, 1.0, pmf(1));
        test_exact(2, 1, 1, 0.5, pmf(0));
        test_exact(2, 1, 1, 0.5, pmf(1));
        test_exact(2, 2, 2, 1.0, pmf(2));
        test_exact(10, 1, 1, 0.9, pmf(0));
        test_exact(10, 1, 1, 0.1, pmf(1));
        test_exact(10, 5, 3, 0.41666666666666666667, pmf(1));
        test_exact(10, 5, 3, 0.083333333333333333333, pmf(3));
    }

    #[test]
    fn test_ln_pmf() {
        let ln_pmf = |arg: u64| move |x: Hypergeometric| x.ln_pmf(arg);
        test_exact(0, 0, 0, 0.0, ln_pmf(0));
        test_exact(1, 1, 1, 0.0, ln_pmf(1));
        test_exact(2, 1, 1, -0.6931471805599453094172, ln_pmf(0));
        test_exact(2, 1, 1, -0.6931471805599453094172, ln_pmf(1));
        test_exact(2, 2, 2, 0.0, ln_pmf(2));
        test_absolute(10, 1, 1, -0.1053605156578263012275, 1e-14, ln_pmf(0));
        test_absolute(10, 1, 1, -2.302585092994045684018, 1e-14, ln_pmf(1));
        test_absolute(10, 5, 3, -0.875468737353899935621, 1e-14, ln_pmf(1));
        test_absolute(10, 5, 3, -2.484906649788000310234, 1e-14, ln_pmf(3));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: Hypergeometric| x.cdf(arg);
        test_exact(2, 1, 1, 0.5, cdf(0));
        test_absolute(10, 1, 1, 0.9, 1e-14, cdf(0));
        test_absolute(10, 5, 3, 0.5, 1e-15, cdf(1));
        test_absolute(10, 5, 3, 11.0 / 12.0, 1e-14, cdf(2));
        test_absolute(10000, 2, 9800, 199.0 / 499950.0, 1e-14, cdf(0));
        test_absolute(10000, 2, 9800, 19799.0 / 499950.0, 1e-12, cdf(1));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: Hypergeometric| x.sf(arg);
        test_exact(2, 1, 1, 0.5, sf(0));
        test_absolute(10, 1, 1, 0.1, 1e-14, sf(0));
        test_absolute(10, 5, 3, 0.5, 1e-15, sf(1));
        test_absolute(10, 5, 3, 1.0 / 12.0, 1e-14, sf(2));
        test_absolute(10000, 2, 9800, 499751. / 499950.0, 1e-10, sf(0));
        test_absolute(10000, 2, 9800, 480151. / 499950.0, 1e-10, sf(1));
    }

    #[test]
    fn test_cdf_arg_too_big() {
        let cdf = |arg: u64| move |x: Hypergeometric| x.cdf(arg);
        test_exact(0, 0, 0, 1.0, cdf(0));
    }

    #[test]
    fn test_cdf_arg_too_small() {
        let cdf = |arg: u64| move |x: Hypergeometric| x.cdf(arg);
        test_exact(2, 2, 2, 0.0, cdf(0));
    }

    #[test]
    fn test_sf_arg_too_big() {
        let sf = |arg: u64| move |x: Hypergeometric| x.sf(arg);
        test_exact(0, 0, 0, 0.0, sf(0));
    }

    #[test]
    fn test_sf_arg_too_small() {
        let sf = |arg: u64| move |x: Hypergeometric| x.sf(arg);
        test_exact(2, 2, 2, 1.0, sf(0));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&create_ok(5, 4, 3), 4);
        test::check_discrete_distribution(&create_ok(3, 2, 1), 2);
    }
}
