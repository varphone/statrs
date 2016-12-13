use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use statistics::*;
use distribution::{Continuous, Distribution};
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

impl Sample<Vec<f64>> for Dirichlet {
    /// Generate random samples from a dirichlet
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> Vec<f64> {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<Vec<f64>> for Dirichlet {
    /// Generate random independent samples from a dirichlet
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> Vec<f64> {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<Vec<f64>> for Dirichlet {
    /// Generate random samples from the dirichlet distribution
    /// using `r` as the source of randomness
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Dirichlet, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> Vec<f64> {
        let n = self.alpha.len();
        let mut samples = vec![0.0; n];
        let mut sum = 0.0;
        for i in 0..n {
            samples[i] = super::gamma::sample_unchecked(r, self.alpha[i], 1.0);
            sum += samples[i];
        }
        for i in 0..n {
            samples[i] /= sum
        }
        samples
    }
}
