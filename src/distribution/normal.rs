use std::f64;
use std::option::Option;
use rand::Rng;
use consts;
use distribution::{Distribution, Univariate, Continuous};
use error::StatsError;
use functions::erf;
use result;

pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> result::Result<Normal> {
        if mean.is_nan() || std_dev < 0.0 {
            return Err(StatsError::BadParams);
        }
        Ok(Normal {
            mu: mean,
            sigma: std_dev,
        })
    }
}

impl Distribution for Normal {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.mu, self.sigma)
    }
}

impl Univariate for Normal {
    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    fn std_dev(&self) -> f64 {
        self.sigma
    }

    fn entropy(&self) -> f64 {
        self.sigma.ln() + consts::LN_SQRT_2PIE
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn median(&self) -> Option<f64> {
        Some(self.mu)
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        Ok(0.5 * erf::erfc((self.mu - x) / (self.sigma * f64::consts::SQRT_2)))
    }
}

impl Continuous for Normal {
    fn mode(&self) -> f64 {
        self.mu
    }

    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        let d = (x - self.mu) / self.sigma;
        (-0.5 * d * d).exp() / (consts::SQRT_2PI * self.sigma)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let d = (x - self.mu) / self.sigma;
        (-0.5 * d * d) - consts::LN_SQRT_2PI - self.sigma.ln()
    }
}

pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    pub fn new(mean: f64, std_dev: f64) -> result::Result<Normal> {
        if mean.is_nan() || std_dev < 0.0 {
            return Err(StatsError::BadParams);
        }
        Ok(Normal {
            mu: mean,
            sigma: std_dev,
        })
    }
}

impl Distribution for LogNormal {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.mu, self.sigma).exp()
    }
}

impl Univariate for LogNormal {
    fn mean(&self) -> f64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp()
    }

    fn variance(&self) -> f64 {
        let sigma2 = self.sigma * self.sigma;
        (sigma2.exp() - 1.0) * (self.mu + self.mu + sigma2).exp()
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        0.5 + self.sigma.ln() + self.mu + consts::LN_SQRT_2PI
    }

    fn skewness(&self) -> f64 {
        let expsigma2 = (self.sigma * self.sigma).exp();
        (expsigma2 + 2.0) * (expsigma2 - 1.0).sqrt()
    }

    fn median(&self) -> Option<f64> {
        Some(self.mu.exp())
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        if x < 0.0 {
            Ok(0.0)
        } else {
            Ok(0.5 * erf::erfc((self.mu - x.ln()) / (self.sigma * f64::consts::SQRT_2)))
        }
    }
}

impl Continuous for LogNormal {
    fn mode(&self) -> f64 {
        (self.mu - self.sigma * self.sigma).exp()
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        match x {
            0.0 => 0.0,
            _ => {
                let d = (x.ln() - self.mu) / self.sigma;
                (-0.5 * d * d).exp() / (x * consts::SQRT_2PI * self.sigma)
            }
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        match x {
            0.0 => f64::NEG_INFINITY,
            _ => {
                let d = (x.ln() - self.mu) / self.sigma;
                (-0.5 * d * d) - consts::LN_SQRT_2PI - (x * self.sigma).ln()
            }
        }
    }
}

/// sample_unchecked draws a sample from a normal distribution using
/// the box-muller algorithm
pub fn sample_unchecked<R: Rng>(r: &mut R, mean: f64, std_dev: f64) -> f64 {
    let mut tuple = polar_transform(r.next_f64(), r.next_f64());
    while !tuple.2 {
        tuple = polar_transform(r.next_f64(), r.next_f64());
    }
    mean + std_dev * tuple.0
}

fn polar_transform(a: f64, b: f64) -> (f64, f64, bool) {
    let v1 = 2.0 * a - 1.0;
    let v2 = 2.0 * b - 1.0;
    let r = v1 * v2 + v2 * v2;
    if r >= 1.0 || r == 0.0 {
        return (0.0, 0.0, false);
    }

    let fac = (-2.0 * r.ln() / r).sqrt();
    (v1 * fac, v2 * fac, true)
}
