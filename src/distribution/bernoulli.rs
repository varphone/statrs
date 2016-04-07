use rand::Rng;
use distribution::{Binomial, Distribution, Univariate, Discrete};
use result;

#[derive(Debug, Clone, PartialEq)]
pub struct Bernoulli {
    b: Binomial
}

impl Bernoulli {
    pub fn new(p: f64) -> result::Result<Bernoulli> {
        match Binomial::new(p, 1) {
            Ok(b) => Ok(Bernoulli{b: b}),
            Err(e) => Err(e)
        }
    }
    
    pub fn p(&self) -> f64 {
        self.b.p()
    }
    
    pub fn n(&self) -> f64 {
        1.0
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

    fn median(&self) -> Option<f64> {
        self.b.median()
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
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