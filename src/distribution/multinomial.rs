use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use distribution::Distribution;
use {Result, StatsError};

/// Implements the [Multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)
/// distribution which is a generalization of the [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Multinomial {
    p: Vec<f64>,
    n: u64,
}

impl Multinomial {
    /// Constructs a new multinomial distribution with probabilities `p`
    /// and `n` number of trials.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is empty, the sum of the elements
    /// in `p` is 0, or any element in `p` is less than 0 or is `f64::NAN`
    ///
    /// # Note
    ///
    /// The elements in `p` do not need to be normalized
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Multinomial;
    ///
    /// let mut result = Multinomial::new(&[0.0, 1.0, 2.0], 3);
    /// assert!(result.is_ok());
    ///
    /// result = Multinomial::new(&[0.0, -1.0, 2.0], 3);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: &[f64], n: u64) -> Result<Multinomial> {
        if !super::internal::is_valid_multinomial(p, true) {
            Err(StatsError::BadParams)
        } else {
            Ok(Multinomial {
                p: p.to_vec(),
                n: n,
            })
        }
    }

    /// Returns the probabilities of the multinomial
    /// distribution as a slice
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Multinomial;
    ///
    /// let n = Multinomial::new(&[0.0, 1.0, 2.0], 3).unwrap();
    /// assert_eq!(n.p(), [0.0, 1.0, 2.0]);
    /// ```
    pub fn p(&self) -> &[f64] {
        &self.p
    }

    /// Returns the number of trials of the multinomial
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Multinomial;
    ///
    /// let n = Multinomial::new(&[0.0, 1.0, 2.0], 3).unwrap();
    /// assert_eq!(n.n(), 3);
    /// ```
    pub fn n(&self) -> u64 {
        self.n
    }
}

impl Sample<Vec<f64>> for Multinomial {
    /// Generate random samples from a multinomial
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> Vec<f64> {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<Vec<f64>> for Multinomial {
    /// Generate random independent samples from a M=multinomial
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> Vec<f64> {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<Vec<f64>> for Multinomial {
    /// Generate random samples from the multinomial distribution
    /// using `r` as the source of randomness
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Multinomial, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Multinomial::new(&[0.0, 1.0, 2.0], 4).unwrap();
    /// print!("{:?}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> Vec<f64> {
        let p_cdf = super::categorical::prob_mass_to_cdf(self.p());
        let mut res = vec![0.0; self.p.len()];
        for _ in 0..self.n {
            let i = super::categorical::sample_unchecked(r, &p_cdf);
            let mut el = unsafe { res.get_unchecked_mut(i as usize) };
            *el = *el + 1.0; 
        }
        res
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use distribution::Multinomial;

    fn try_create(p: &[f64], n: u64) -> Multinomial {
        let dist = Multinomial::new(p, n);
        assert!(dist.is_ok());
        dist.unwrap()
    }

    fn create_case(p: &[f64], n: u64) {
        let dist = try_create(p, n);
        assert_eq!(dist.p(), p);
        assert_eq!(dist.n(), n);
    }

    fn bad_create_case(p: &[f64], n: u64) {
        let dist = Multinomial::new(p, n);
        assert!(dist.is_err());
    }

    #[test]
    fn test_create() {
        create_case(&[1.0, 1.0, 1.0], 4);
        create_case(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 4);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(&[-1.0, 1.0], 4);
        bad_create_case(&[0.0, 0.0], 4);
    }
}