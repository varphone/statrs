use std::cmp;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use function::factorial;
use statistics::*;
use distribution::{Univariate, Discrete, Distribution};
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
    draws: u64,
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
    pub fn new(population: u64, successes: u64, draws: u64) -> Result<Hypergeometric> {
        if successes > population || draws > population {
            Err(StatsError::BadParams)
        } else {
            Ok(Hypergeometric {
                population: population,
                successes: successes,
                draws: draws,
            })
        }
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
}

impl Sample<f64> for Hypergeometric {
    /// Generate a random sample from a hypergeometric
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Hypergeometric {
    /// Generate a random independent sample from a hypergeometric
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for Hypergeometric {
    /// Generates a random sample from the hypergeometric distribution
    /// using `r` as the source of randomness
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Hypergeometric, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Hypergeometric::new(10, 5, 3).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        let mut population = self.population as f64;
        let mut successes = self.successes as f64;
        let mut draws = self.draws;
        let mut x = 0.0;
        loop {
            let p = successes / population;
            let next = r.next_f64();
            if next < p {
                x += 1.0;
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

impl Univariate<u64, f64> for Hypergeometric {
    /// Calculates the cumulative distribution function for the hypergeometric
    /// distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x < n + K - N` or `x >= min(K, n)`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// TODO
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        assert!(x >= self.min() as f64,
                format!("{}", StatsError::ArgGteArg("x", "n + K - N")));
        assert!(x < self.max() as f64,
                format!("{}", StatsError::ArgLtArg("x", "min(K, n)")));
        let k = x.floor() as u64;
        let ln_denom = factorial::ln_binomial(self.population, self.draws);
        (0..k + 1).fold(0.0, |acc, i| {
            acc +
            (factorial::ln_binomial(self.successes, i) +
             factorial::ln_binomial(self.population - self.successes, self.draws - i) -
             ln_denom)
                .exp()
        })
    }
}

impl Min<u64> for Hypergeometric {
    /// Returns the minimum value in the domain of the
    /// hypergeometric distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 -
    /// ```
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
    /// ```ignore
    /// 2^63 - 1
    /// ```
    fn max(&self) -> u64 {
        cmp::min(self.successes, self.draws)
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use statistics::*;
    use distribution::{Univariate, Discrete, Hypergeometric};

    fn try_create(population: u64, successes: u64, draws: u64) -> Hypergeometric {
        let n = Hypergeometric::new(population, successes, draws);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(population: u64, successes: u64, draws: u64) {
        let n = try_create(population, successes, draws);
        assert_eq!(population, n.population());
        assert_eq!(successes, n.successes());
        assert_eq!(draws, n.draws());
    }

    fn bad_create_case(population: u64, successes: u64, draws: u64) {
        let n = Hypergeometric::new(population, successes, draws);
        assert!(n.is_err());
    }

    fn get_value<T, F>(population: u64, successes: u64, draws: u64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Hypergeometric) -> T
    {
        let n = try_create(population, successes, draws);
        eval(n)
    }

    fn test_case<T, F>(population: u64, successes: u64, draws: u64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Hypergeometric) -> T
    {
        let x = get_value(population, successes, draws, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(population: u64, successes: u64, draws: u64, expected: f64, acc: f64, eval: F)
        where F: Fn(Hypergeometric) -> f64
    {
        let x = get_value(population, successes, draws, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(0, 0, 0);
        create_case(1, 1, 1,);
        create_case(2, 1, 1);
        create_case(2, 2, 2);
        create_case(10, 1, 1);
        create_case(10, 5, 3);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(2, 3, 2);
        bad_create_case(10, 5, 20);
        bad_create_case(0, 1, 1);
    }

    #[test]
    fn test_min() {
        test_case(0, 0, 0, 0, |x| x.min());
        test_case(1, 1, 1, 1, |x| x.min());
        test_case(2, 1, 1, 0, |x| x.min());
        test_case(2, 2, 2, 2, |x| x.min());
        test_case(10, 1, 1, 0, |x| x.min());
        test_case(10, 5, 3, 0, |x| x.min());
    }

    #[test]
    fn test_max() {
        test_case(0, 0, 0, 0, |x| x.max());
        test_case(1, 1, 1, 1, |x| x.max());
        test_case(2, 1, 1, 1, |x| x.max());
        test_case(2, 2, 2, 2, |x| x.max());
        test_case(10, 1, 1, 1, |x| x.max());
        test_case(10, 5, 3, 3, |x| x.max());
    }

    #[test]
    fn test_cdf() {
        test_case(2, 1, 1, 0.5, |x| x.cdf(0.3));
        test_almost(10, 1, 1, 0.9, 1e-14, |x| x.cdf(0.3));
        test_almost(10, 5, 3, 0.5, 1e-15, |x| x.cdf(1.1));
        test_almost(10, 5, 3, 11.0 / 12.0, 1e-14, |x| x.cdf(2.0));
        test_almost(10000, 2, 9800, 199.0 / 499950.0, 1e-14, |x| x.cdf(0.0));
        test_almost(10000, 2, 9800, 199.0 / 499950.0, 1e-14, |x| x.cdf(0.5));
        test_almost(10000, 2, 9800, 19799.0 / 499950.0, 1e-12, |x| x.cdf(1.5));
    }

    #[test]
    #[should_panic]
    fn test_cdf_arg_too_big() {
        get_value(0, 0, 0, |x| x.cdf(0.5));
    }

    #[test]
    #[should_panic]
    fn test_cdf_arg_too_small() {
        get_value(2, 2, 2, |x| x.cdf(0.0));
    }
}