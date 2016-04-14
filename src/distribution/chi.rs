use std::f64;
use rand::Rng;
use consts;
use distribution::{Gamma, Distribution, Univariate, Continuous};
use function;
use result;

#[derive(Debug, Clone, PartialEq)]
pub struct Chi {
    k: f64
}

impl Chi {
    pub fn new(freedom: f64) -> result::Result<Chi> {
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
        0.0
    }
}

impl Univariate for Chi {
    fn mean(&self) -> f64 {
        consts::SQRT_2 * functions::gamma((self.k + 1.0) / 2.0) / functions::gamma(self.k / 2.0)
    }

    fn variance(&self) -> f64 {
        self.k - self.mean() * self.mean()
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        functions::ln_gamma(self.k / 2.0) + (self.k - (2.0f64).ln() - (self.k - 1.0) * functions::digamma(self.k / 2.0)) / 2.0
    }

    fn skewness(&self) -> f64 {
        let sigma = self.std_dev();
        self.mean() * (1.0 - 2.0 * sigma * sigma) / (sigma * sigma * sigma)
    }

    fn median(&self) -> f64 {
        panic!("Unsupported")
    }

    fn cdf(&self, x: f64) -> f64 {
        if x == f64::INFINITY || self.k === f64::INFINITY {
            1.0
        } else {
            functions::gamma_lr(self.k / 2.0, x * x / 2.0)
        }
    }
}

impl Continuous for Chi {
    fn mode(&self) -> f64 {
        0.0
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        0.0
    }

    fn pdf(&self, x: f64) -> f64 {
       0.0
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        0.0
    }
}