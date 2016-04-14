use std::f64;
use std::option::Option;
use rand::Rng;
use consts;
use distribution::{Distribution, Univariate, Continuous};
use error::StatsError;
use result;

#[derive(Debug, Clone, PartialEq)]
pub struct Exponential {
    rate: f64
}

impl Exponential {
    pub fn new(rate: f64) -> result::Result<Exponential> {
        if rate.is_nan() || rate <= 0.0 {
            Err(StatsError::BadParams);
        } else {
            Ok(Exponential{rate: rate})
        }
    }
    
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Distribution for Exponential {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        0.0
    }
}

impl Univariate for Exponential {
    fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }

    fn std_dev(&self) -> f64 {
        1.0 / self.rate
    }

    fn entropy(&self) -> f64 {
        1.0 - self.rate.ln()
    }

    fn skewness(&self) -> f64 {
        2.0
    }

    fn median(&self) -> Option<f64> {
        Some(consts::LN_2 / self.rate)
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        if x < 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - (-self.rate * x).exp())
        }
    }
}

impl Continuous for Exponential {
    fn mode(&self) -> f64 {
        0.0
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.rate.ln() - self.rate * x
    }
}