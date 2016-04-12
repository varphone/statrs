use std::f64;
use std::option::Option;
use rand::Rng;
use distribution::{Distribution, Univariate, Discrete};
use error::StatsError;
use result;

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteUniform {
    min: i64,
    max: i64
} 

impl DiscreteUniform {
    pub fn new(min: i64, max: i64) -> result::Result<DiscreteUniform> {
        if max > min {
            return Err(StatsError::BadParams);
        }
        Ok(DiscreteUniform{
            min: min,
            max: max
        })
    }
}

impl Distribution for DiscreteUniform {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        r.gen_range(self.min, self.max + 1) as f64
    }
}

impl Univariate for DiscreteUniform {
     fn mean(&self) -> f64 {
        (self.min + self.max) as f64 / 2.0
    }

    fn variance(&self) -> f64 {
        let diff = (self.max - self.min) as f64;
        ((diff + 1.0) * (diff + 1.0) - 1.0) / 12.0
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        let diff = (self.max - self.min) as f64;
        (diff + 1.0).ln()
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn median(&self) -> Option<f64> {
        Some((self.min + self.max) as f64 / 2.0)
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        if x < self.min as f64 {
            return Ok(0.0);
        }
        if x >= self.max as f64 {
            return Ok(1.0);
        }
        let lower = self.min as f64;
        let upper = self.max as f64;
        let ans = (x.floor() - lower + 1.0) / (upper - lower + 1.0);
        if x > 1.0 {
            return Ok(1.0);
        }
        Ok(ans)
    }
}

impl Discrete for DiscreteUniform {
    fn mode(&self) -> i64 {
        ((self.min + self.max) as f64 / 2.0).floor() as i64
    }

    fn min(&self) -> i64 {
        self.min
    }

    fn max(&self) -> i64 {
        self.max
    }

    fn pmf(&self, x: i64) -> f64 {
        if x >= self.min && x <= self.max {
            return 1.0 / (self.max - self.min + 1) as f64
        }
        0.0
    }

    fn ln_pmf(&self, x: i64) -> f64 {
        if x >= self.min && x <= self.max {
            return -((self.max - self.min + 1) as f64).ln()
        }
        f64::NEG_INFINITY
    }
}