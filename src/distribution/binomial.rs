use std::f64;
use rand::Rng;
use distribution::{Distribution, Univariate, Discrete};
use error::StatsError;
use function::beta;
use function::factorial;
use result;

pub struct Binomial {
    p: f64,
    n: i64,
}

impl Binomial {
    pub fn new(p: f64, n: i64) -> result::Result<Binomial> {
        if p < 0.0 || p > 1.0 || n < 0 {
            return Err(StatsError::BadParams);
        }
        Ok(Binomial { p: p, n: n })
    }
    
    pub fn p(&self) -> f64 {
        self.p
    }
    
    pub fn n(&self) -> i64 {
        self.n
    }
}

impl Distribution for Binomial {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        (0..self.n).fold(0.0, |acc, _| {
            let n = r.next_f64();
            return if n < self.p {
                acc + 1.0
            } else {
                acc
            };
        })
    }
}

impl Univariate for Binomial {
    fn mean(&self) -> f64 {
        self.p * self.n as f64
    }

    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p) * self.n as f64
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        match self.p {
            0.0 | 1.0 => 0.0,
            _ => {
                (0..self.n).fold(0.0, |acc, x| {
                    let p = self.pmf(x);
                    acc - p * p.ln()
                })
            }
        }
    }

    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * self.p) / (self.n as f64 * self.p * (1.0 - self.p)).sqrt()
    }

    fn median(&self) -> Option<f64> {
        Some((self.p * self.n as f64).floor())
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        if x < 0.0 {
            return Ok(0.0);
        }
        if x > self.n as f64 {
            return Ok(1.0);
        }
        let k = x.floor();
        beta::beta_reg(self.n as f64 - k, k + 1.0, 1.0 - self.p)
    }
}

impl Discrete for Binomial {
    fn mode(&self) -> i64 {
        match self.p {
            0.0 => 0,
            1.0 => self.n,
            _ => ((self.n as f64 + 1.0) * self.p).floor() as i64,
        }
    }

    fn min(&self) -> i64 {
        0
    }

    fn max(&self) -> i64 {
        self.n
    }

    fn pmf(&self, x: i64) -> f64 {
        if x < 0 || x > self.n {
            return 0.0;
        }
        if self.p == 0.0 {
            return if x == 0 {
                1.0
            } else {
                0.0
            };
        }
        if self.p == 1.0 {
            return if x == self.n {
                1.0
            } else {
                0.0
            };
        }
        (factorial::ln_binomial(self.n as u64, x as u64) + x as f64 * self.p.ln() +
         (self.n - x) as f64 * (1.0 - self.p).ln())
            .exp()
    }

    fn ln_pmf(&self, x: i64) -> f64 {
        if x < 0 || x > self.n {
            return f64::NEG_INFINITY;
        }
        if self.p == 0.0 {
            return if x == 0 {
                0.0
            } else {
                f64::NEG_INFINITY
            };
        }
        if self.p == 1.0 {
            return if x == self.n {
                0.0
            } else {
                f64::NEG_INFINITY
            };
        }
        factorial::ln_binomial(self.n as u64, x as u64) + x as f64 * self.p.ln() +
        (self.n - x) as f64 * (1.0 - self.p).ln()
    }
}

#[cfg(tests)]
mod test {
    use std::f64;
    use std::option::Option;
    use distribution::{Univariate, Continuous};
    use prec;
    use result;
    use super::Binomial;
    
    fn try_create(p: f64, n: i64) -> Binomial {
        let n = Binomial::new(p, n);
        assert!(n.is_ok());
        n.unwrap()
    }
    
    fn create_case(p: f64, n: i64) {
        let n = try_create(p, n);
        assert_eq!(p, n.p());
    }
    
    fn bad_create_case(p: f64, n: i64) {
        let n = Binomial::new(p, n);
        assert!(n.is_err());
    }
    
    #[test]
    fn test_create() {
        create_case(0.0, 4);
        create_case(0.3, 3);
        create_case(1.0, 2);
    }
    
    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1);
        bad_create_case(-1.0, 1);
        bad_create_case(2.0, 1);
        bad_create_case(0.3, -2);
    }
}