use std::f64;
use std::option::Option;
use rand::Rng;
use distribution::{Distribution, Univariate, Continuous};
use error::StatsError;
use result;

pub struct Uniform {
    min: f64,
    max: f64,
}

impl Uniform {
    fn new(min: f64, max: f64) -> result::Result<Uniform> {
        if min > max {
            return Err(StatsError::BadParams);
        }
        Ok(Uniform {
            min: min,
            max: max,
        })
    }
}

impl Distribution for Uniform {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        self.min + r.next_f64() * (self.max - self.min)
    }
}

impl Univariate for Uniform {
    fn mean(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    fn variance(&self) -> f64 {
        (self.max - self.min) * (self.max - self.min) / 12.0
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        (self.max - self.min).ln()
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn median(&self) -> Option<f64> {
        Some((self.min + self.max) / 2.0)
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        return if x <= self.min {
            Ok(0.0)
        } else if x >= self.max {
            Ok(1.0)
        } else {
            Ok((x - self.min) / (self.max - self.min))
        };
    }
}

impl Continuous for Uniform {
    fn mode(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }

    fn pdf(&self, x: f64) -> f64 {
        return if x < self.min || x > self.max {
            0.0
        } else {
            1.0 / (self.max - self.min)
        };
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        return if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            -(self.max - self.min).ln()
        };
    }
}
