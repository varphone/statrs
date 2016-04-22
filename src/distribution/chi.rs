use std::f64;
use rand::Rng;
use function::gamma;
use result::Result;
use super::{Gamma, Distribution, Univariate, Continuous};
use super::normal;

#[derive(Debug, Clone, PartialEq)]
pub struct Chi {
    k: f64
}

impl Chi {
    pub fn new(freedom: f64) -> Result<Chi> {
        if freedom.is_nan() || freedom <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Chi{k: freedom})
        }
    }
    
    pub fn freedom(&self) -> f64 {
        self.k
    }
}

impl Distribution for Chi {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        (0.0..self.k).fold(0.0, |acc, _| {
            acc + normal::sample_unchecked(r, 0.0, 1.0).powf(2.0)
        }).sqrt()
    }
}

impl Univariate for Chi {
    fn mean(&self) -> f64 {
        f64::consts::SQRT_2 * gamma::gamma((self.k + 1.0) / 2.0) / gamma::gamma(self.k / 2.0)
    }

    fn variance(&self) -> f64 {
        self.k - self.mean() * self.mean()
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        gamma::ln_gamma(self.k / 2.0) + (self.k - (2.0f64).ln() - (self.k - 1.0) * gamma::digamma(self.k / 2.0)) / 2.0
    }

    fn skewness(&self) -> f64 {
        let sigma = self.std_dev();
        self.mean() * (1.0 - 2.0 * sigma * sigma) / (sigma * sigma * sigma)
    }

    fn median(&self) -> f64 {
        unimplemented!()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x == f64::INFINITY || self.k === f64::INFINITY {
            1.0
        } else {
            gamma::gamma_lr(self.k / 2.0, x * x / 2.0)
        }
    }
}

impl Continuous for Chi {
    fn mode(&self) -> f64 {
        if self.k < 1.0 {
            panic!("Cannot calculate Chi distribution mode for freedom < 1");
        }
        (self.k - 1.0).sqrt()
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
       match (self.k, x) {
           (f64::INFINITY, _) | (_, f64::INFINITY) | (_, 0.0) => 0.0,
           (_, _) if self.k > 160.0 => self.ln_pdf(x),
           (_, _) => {
               (2.0f64).powf(1.0 - self.k / 2.0) * x.powf(self.k - 1.0) 
               * (-x * x / 2.0).exp() / gamma::gamma(self.k / 2.0)
           }
       }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        match (self.k, x) {
            (f64::INFINITY, _) | (_, f64::INFINITY) | (_, 0.0) => f64::NEG_INFINITY,
            (_, _) => {
                (1.0 - self.k / 2.0) * (2.0f64).ln() + ((self.k - 1.0) * x.ln()) 
                - x * x / 2.0 - gamma::ln_gamma(self.k / 2.0)
            }
        }
    }
}