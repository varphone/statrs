use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use statistics::*;
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

impl Mean<Vec<f64>> for Multinomial {
    /// Returns the mean of the multinomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// n * p_i for i in 1..k
    /// ```
    ///
    /// where `n` is the number of trials, `p_i` is the `i`th probability,
    /// and `k` is the total number of probabilities
    fn mean(&self) -> Vec<f64> {
        self.p.iter().map(|x| x * self.n as f64).collect()
    }
}

impl Variance<Vec<f64>> for Multinomial {
    /// Returns the variance of the multinomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// n * p_i * (1 - p_1)
    /// ```
    ///
    /// where `n` is the number of trials, `p_i` is the `i`th probability,
    /// and `k` is the total number of probabilities
    fn variance(&self) -> Vec<f64> {
        self.p.iter().map(|x| x * self.n as f64 * (1.0 - x)).collect()
    }

    /// Returns the standard deviation of the multinomial distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(n * p_i * (1 - p_1))
    /// ```
    ///
    /// where `n` is the number of trials, `p_i` is the `i`th probability,
    /// and `k` is the total number of probabilities
    fn std_dev(&self) -> Vec<f64> {
        self.variance().iter().map(|x| x.sqrt()).collect()
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use statistics::*;
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

    fn test_case<F>(p: &[f64], n: u64, expected: &[f64], eval: F)
        where F: Fn(Multinomial) -> Vec<f64>
    {
        let dist = try_create(p, n);
        let x = eval(dist);
        assert_eq!(*expected, *x);
    }

    fn test_almost<F>(p: &[f64], n: u64, expected: &[f64], acc: f64, eval: F)
        where F: Fn(Multinomial) -> Vec<f64>
    {
        let dist = try_create(p, n);
        let x = eval(dist);
        assert_eq!(expected.len(), x.len());
        for i in 0..expected.len() {
            assert_almost_eq!(expected[i], x[i], acc);
        }
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

    #[test]
    fn test_mean() {
        test_case(&[0.3, 0.7], 5, &[1.5, 3.5], |x| x.mean());
        test_case(&[0.1, 0.3, 0.6], 10, &[1.0, 3.0, 6.0], |x| x.mean());
        test_case(&[0.15, 0.35, 0.3, 0.2], 20, &[3.0, 7.0, 6.0, 4.0], |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_almost(&[0.3, 0.7], 5, &[1.05, 1.05], 1e-15, |x| x.variance());
        test_almost(&[0.1, 0.3, 0.6], 10, &[0.9, 2.1, 2.4], 1e-15, |x| x.variance());
        test_almost(&[0.15, 0.35, 0.3, 0.2], 20, &[2.55, 4.55, 4.2, 3.2], 1e-15, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_almost(&[0.3, 0.7], 5, &[1.05f64.sqrt(), 1.05f64.sqrt()], 1e-15, |x| x.std_dev());
        test_almost(&[0.1, 0.3, 0.6], 10, &[0.9f64.sqrt(), 2.1f64.sqrt(), 2.4f64.sqrt()], 1e-15, |x| x.std_dev());
        test_almost(&[0.15, 0.35, 0.3, 0.2], 20, &[2.55f64.sqrt(), 4.55f64.sqrt(), 4.2f64.sqrt(), 3.2f64.sqrt()], 1e-15, |x| x.std_dev());
    }
}