use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use result::Result;
use super::{Gamma, Distribution, Univariate, Continuous};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ChiSquared {
    freedom: f64,
    g: Gamma,
}

impl ChiSquared {
    pub fn new(freedom: f64) -> Result<ChiSquared> {
        Gamma::new(freedom / 2.0, 0.5).map(|g| {
            ChiSquared {
                freedom: freedom,
                g: g,
            }
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    pub fn shape(&self) -> f64 {
        self.g.shape()
    }

    pub fn rate(&self) -> f64 {
        self.g.rate()
    }
}

impl Sample<f64> for ChiSquared {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for ChiSquared {
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution for ChiSquared {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        self.g.sample(r)
    }
}

impl Univariate for ChiSquared {
    fn mean(&self) -> f64 {
        self.g.mean()
    }

    fn variance(&self) -> f64 {
        self.g.variance()
    }

    fn std_dev(&self) -> f64 {
        self.g.std_dev()
    }

    fn entropy(&self) -> f64 {
        self.g.entropy()
    }

    fn skewness(&self) -> f64 {
        self.g.skewness()
    }

    fn median(&self) -> f64 {
        unimplemented!()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.g.cdf(x)
    }
}

impl Continuous for ChiSquared {
    fn mode(&self) -> f64 {
        self.g.mode()
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        self.g.pdf(x)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.g.ln_pdf(x)
    }
}
