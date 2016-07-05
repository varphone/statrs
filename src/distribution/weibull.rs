use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use consts;
use error::StatsError;
use function::{gamma, stable};
use result::Result;
use super::{Distribution, Univariate, Continuous};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Weibull {
    shape: f64,
    scale: f64,
    scale_pow_shape_inv: f64,
}

impl Weibull {
    pub fn new(shape: f64, scale: f64) -> Result<Weibull> {
        if shape <= 0.0 || scale <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Weibull {
                shape: shape,
                scale: scale,
                scale_pow_shape_inv: scale.powf(-shape),
            })
        }
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Sample<f64> for Weibull {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Weibull {
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution for Weibull {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        let x = r.next_f64();
        self.scale * (-x.ln()).powf(1.0 / self.shape)
    }
}

impl Univariate for Weibull {
    fn mean(&self) -> f64 {
        self.scale * gamma::gamma(1.0 + 1.0 / self.shape)
    }

    fn variance(&self) -> f64 {
        self.scale * self.scale * gamma::gamma(1.0 + 2.0 / self.shape) - self.mean() * self.mean()
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        consts::EULER_MASCHERONI * (1.0 - 1.0 / self.shape) + (self.scale / self.shape).ln() + 1.0
    }

    fn skewness(&self) -> f64 {
        let mu = self.mean();
        let sigma = self.std_dev();
        let sigma2 = sigma * sigma;
        let sigma3 = sigma2 * sigma;
        self.scale * self.scale * self.scale * gamma::gamma(1.0 + 3.0 / self.shape) -
        3.0 * sigma2 * mu - (mu * mu * mu) / sigma3
    }

    fn median(&self) -> f64 {
        self.scale * f64::consts::LN_2.powf(1.0 / self.shape)
    }

    fn cdf(&self, x: f64) -> f64 {
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        -stable::exp_minus_one(x.powf(self.shape) * self.scale_pow_shape_inv)
    }
}

impl Continuous for Weibull {
    fn mode(&self) -> f64 {
        if self.shape <= 1.0 {
            0.0
        } else {
            self.scale * ((self.shape - 1.0) / self.shape).powf(1.0 / self.shape)
        }
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        match (x, self.shape) {
            (0.0, 1.0) => self.shape / self.scale,
            (_, _) if x >= 0.0 => {
                self.shape * (x / self.scale).powf(self.shape - 1.0) *
                (-(x.powf(self.shape)) * self.scale_pow_shape_inv).exp() /
                self.scale
            }
            (_, _) => 0.0,
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        match (x, self.shape) {
            (0.0, 1.0) => self.shape.ln() - self.scale.ln(),
            (_, _) if x >= 0.0 => {
                self.shape.ln() + (self.shape - 1.0) * (x / self.scale).ln() -
                x.powf(self.shape) * self.scale_pow_shape_inv - self.scale.ln()
            }
            (_, _) => f64::NEG_INFINITY, 
        }
    }
}
