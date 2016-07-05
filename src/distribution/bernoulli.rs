use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use result::Result;
use super::{Binomial, Distribution, Univariate, Discrete};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Bernoulli {
    b: Binomial,
}

impl Bernoulli {
    pub fn new(p: f64) -> Result<Bernoulli> {
        Binomial::new(p, 1).map(|b| Bernoulli { b: b })
    }

    pub fn p(&self) -> f64 {
        self.b.p()
    }

    pub fn n(&self) -> f64 {
        1.0
    }
}

impl Sample<f64> for Bernoulli {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Bernoulli {
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution for Bernoulli {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        self.b.sample(r)
    }
}

impl Univariate for Bernoulli {
    fn mean(&self) -> f64 {
        self.b.mean()
    }

    fn variance(&self) -> f64 {
        self.b.variance()
    }

    fn std_dev(&self) -> f64 {
        self.b.std_dev()
    }

    fn entropy(&self) -> f64 {
        self.b.entropy()
    }

    fn skewness(&self) -> f64 {
        self.b.skewness()
    }

    fn median(&self) -> f64 {
        self.b.median()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.b.cdf(x)
    }
}

impl Discrete for Bernoulli {
    fn mode(&self) -> i64 {
        self.b.mode()
    }

    fn min(&self) -> i64 {
        0
    }

    fn max(&self) -> i64 {
        1
    }

    fn pmf(&self, x: i64) -> f64 {
        self.b.pmf(x)
    }

    fn ln_pmf(&self, x: i64) -> f64 {
        self.b.ln_pmf(x)
    }
}
