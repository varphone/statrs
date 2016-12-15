use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use function::gamma;
use statistics::*;
use distribution::{Continuous, Distribution};
use result::Result;
use error::StatsError;
use prec;

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

    fn alpha_sum(&self) -> f64 {
        self.alpha.iter().fold(0.0, |acc, x| acc + x)
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

impl Mean<Vec<f64>> for Dirichlet {
    /// Returns the means of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// α_i / α_0
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn mean(&self) -> Vec<f64> {
        let sum = self.alpha_sum();
        self.alpha.iter().map(|x| x / sum).collect()
    }
}

impl Variance<Vec<f64>> for Dirichlet {
    /// Returns the variances of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (α_i * (α_0 - α_i)) / (α_0^2 * (α_0 + 1))
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn variance(&self) -> Vec<f64> {
        let sum = self.alpha_sum();
        self.alpha.iter().map(|x| x * (sum - x) / (sum * sum * (sum + 1.0))).collect()
    }

    /// Returns the variances of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt((α_i * (α_0 - α_i)) / (α_0^2 * (α_0 + 1)))
    /// ```
    ///
    /// for the `i`th element where `α_i` is the `i`th concentration parameter
    /// and `α_0` is the sum of all concentration parameters
    fn std_dev(&self) -> Vec<f64> {
        self.variance().iter().map(|x| x.sqrt()).collect()
    }
}

impl Entropy<f64> for Dirichlet {
    /// Returns the entropy of the dirichlet distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(B(α)) - (K - α_0)ψ(α_0) - ∑((α_i - 1)ψ(α_i))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = ∏(Γ(α_i)) / Γ(∑(α_i))
    /// ```
    ///
    /// `α_0` is the sum of all concentration parameters,
    /// `K` is the number of concentration parameters, `ψ` is the digamma function, `α_i`
    /// is the `i`th concentration parameter, and `∑` is the sum from `1` to `K`
    fn entropy(&self) -> f64 {
        let sum = self.alpha_sum();
        let num = self.alpha.iter().fold(0.0, |acc, &x| acc + (x - 1.0) * gamma::digamma(x));
        gamma::ln_gamma(sum) + (sum - self.alpha.len() as f64) * gamma::digamma(sum) - num
    }
}

impl<'a> Continuous<&'a [f64], f64> for Dirichlet {
    /// Calculates the probabiliy density function for the dirichlet distribution
    /// with given `x`'s corresponding to the concentration parameters for this
    /// distribution
    ///
    /// # Panics
    ///
    /// If any element in `x` is not in `(0, 1)` or if `x` is not the same length
    /// as the vector of concentration parameters for this distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / B(α)) * ∏(x_i^(α_i - 1))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = ∏(Γ(α_i)) / Γ(∑(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `∏` is the product from `1` to `K`, `∑` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn pdf(&self, x: &[f64]) -> f64 {
        self.ln_pdf(x).exp()
    }

    /// Calculates the log probabiliy density function for the dirichlet distribution
    /// with given `x`'s corresponding to the concentration parameters for this
    /// distribution
    ///
    /// # Panics
    ///
    /// If any element in `x` is not in `(0, 1)` or if `x` is not the same length
    /// as the vector of concentration parameters for this distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((1 / B(α)) * ∏(x_i^(α_i - 1)))
    /// ```
    ///
    /// where
    ///
    /// ```ignore
    /// B(α) = ∏(Γ(α_i)) / Γ(∑(α_i))
    /// ```
    ///
    /// `α` is the vector of concentration parameters, `α_i` is the `i`th
    /// concentration parameter, `x_i` is the `i`th argument corresponding to
    /// the `i`th concentration parameter, `Γ` is the gamma function,
    /// `∏` is the product from `1` to `K`, `∑` is the sum from `1` to `K`,
    /// and `K` is the number of concentration parameters
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        assert!(self.alpha.len() == x.len(),
                format!("{}", StatsError::ContainersMustBeSameLength));

        let (term, sum_xi, sum_alpha) = x.iter()
            .enumerate()
            .map(|pair| (pair.1, self.alpha[pair.0]))
            .fold((0.0, 0.0, 0.0), |acc, pair| {
                assert!(*pair.0 > 0.0 && *pair.0 < 1.0,
                        format!("{}", StatsError::ArgIntervalExcl("x", 0.0, 1.0)));

                (acc.0 + (pair.1 - 1.0) * pair.0.ln() - gamma::ln_gamma(pair.1),
                 acc.1 + pair.0,
                 acc.2 + pair.1)
            });

        if !prec::almost_eq(sum_xi, 1.0, 1e-8) {
            0.0
        } else {
            term + gamma::ln_gamma(sum_alpha)
        }
    }
}