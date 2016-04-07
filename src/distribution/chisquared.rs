use std::f64;
use std::option::Option;
use rand::Rng;
use distribution::{Gamma, Distribution, Univariate, Continuous};
use result;

pub struct ChiSquared {
    k: f64,
    g: Gamma
}

impl ChiSquared {
    pub fn new(k: f64) -> result::Result<ChiSquared> {
        match Gamma::new(k / 2.0, 0.5) {
            Ok(g) => Ok(ChiSquared{k: k, g: g}),
            Err(e) => Err(e)
        }
    }
    
    pub fn freedom(&self) -> f64 {
        self.k
    }
    
    pub fn shape(&self) -> f64 {
        self.g.shape()
    }
    
    pub fn rate(&self) -> f64 {
        self.g.rate()
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

    fn median(&self) -> Option<f64> {
        None
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
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