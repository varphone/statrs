use std::f64;
use std::i64;
use rand::Rng;
use error::StatsError;
use function::factorial;
use function::gamma;
use result::Result;
use super::{Distribution, Univariate, Discrete};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Poisson {
    lambda: f64
}

impl Poisson {
    pub fn new(lambda: f64) -> Result<Poisson> {
        if lambda.is_nan() || lambda < 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Poisson{lambda: lambda})
        }
    }
    
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution for Poisson {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        0.0
    }
}

impl Univariate for Poisson {
    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        self.lambda
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        0.5 * (2.0 * f64::consts::PI * f64::consts::E * self.lambda).ln()
        - 1.0 / (12.0 * self.lambda)
        - 1.0 / (24.0 * self.lambda * self.lambda)
        - 19.0 / (360.0 * self.lambda * self.lambda * self.lambda)
    }

    fn skewness(&self) -> f64 {
       1.0 / self.lambda.sqrt()
    }

    fn median(&self) -> f64 {
        (self.lambda + 1.0 / 3.0 - 0.02 / self.lambda).floor()
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 - gamma::gamma_lr(x + 1.0, self.lambda).unwrap()
    }
}

impl Discrete for Poisson {
    fn mode(&self) -> i64 {
        self.lambda.floor() as i64
    }

    fn min(&self) -> i64 {
        0
    }

    fn max(&self) -> i64 {
        i64::MAX
    }

    fn pmf(&self, x: i64) -> f64 {
        if x < 0 {
            panic!("{}", StatsError::ArgNotNegative("x"));
        }
        (-self.lambda + x as f64 * self.lambda.ln()).exp() - factorial::ln_factorial(x as u64)
    }

    fn ln_pmf(&self, x: i64) -> f64 {
        if x < 0 {
            panic!("{}", StatsError::ArgNotNegative("x"));
        }
        -self.lambda + x as f64 * self.lambda.ln() - factorial::ln_factorial(x as u64)
    }
}