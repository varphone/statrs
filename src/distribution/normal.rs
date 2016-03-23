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
        if mean.is_nan() || std_dev.is_nan() || std_dev < 0.0 {
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
    pub fn new(mean: f64, std_dev: f64) -> result::Result<LogNormal> {
        if mean.is_nan() || std_dev.is_nan() || std_dev < 0.0 {
            return Err(StatsError::BadParams);
        }
        Ok(LogNormal {
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

#[cfg(test)]
mod test {
    use std::f64;
    use std::option::Option;
    use distribution::{Univariate, Continuous};
    use prec;
    use result;
    use super::{Normal, LogNormal};
    
    fn try_create(mean: f64, std_dev: f64) -> Normal {
        let n = Normal::new(mean, std_dev);
        assert!(n.is_ok());
        n.unwrap()
    }
    
    fn create_case(mean: f64, std_dev: f64) {
        let n = try_create(mean, std_dev);
        assert_eq!(mean, n.mean());
        assert_eq!(std_dev, n.std_dev());
    }
    
    fn bad_create_case(mean: f64, std_dev: f64) {
        let n = Normal::new(mean, std_dev);
        assert!(n.is_err());
    }
    
    fn test_case<F>(mean: f64, std_dev: f64, expected: f64, eval: F) 
        where F : Fn(Normal) -> f64 {
    
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert_eq!(expected, x);        
    }
    
    fn test_almost<F>(mean: f64, std_dev: f64, expected: f64, acc: f64, eval: F) 
        where F : Fn(Normal) -> f64 {
        
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));    
    }
    
    fn test_optional<F>(mean: f64, std_dev: f64, expected: f64, eval: F)
        where F : Fn(Normal) -> Option<f64> {
    
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert!(x.is_some());
        
        let v = x.unwrap();
        assert_eq!(expected, v);   
    }
    
    #[test]
    fn test_create() {
        create_case(0.0, 0.0);
        create_case(10.0, 0.1);
        create_case(-5.0, 1.0);
        create_case(0.0, 10.0);
        create_case(10.0, 100.0);
        create_case(-5.0, f64::INFINITY);
    }
    
    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, -1.0);
    }
    
    #[test]
    fn test_entropy() {
        // note: mean is irrelevant to the entropy calculation,
        // ergo, all the test cases are instantiated with the
        // same mean
        test_almost(0.0, -0.0, f64::NEG_INFINITY, 1e-15, |x| x.entropy());
        test_almost(0.0, 0.0, f64::NEG_INFINITY, 1e-15, |x| x.entropy());
        test_almost(0.0, 0.1, -0.8836465597893729422377, 1e-15, |x| x.entropy());
        test_almost(0.0, 1.0, 1.41893853320467274178, 1e-15, |x| x.entropy());
        test_almost(0.0, 10.0, 3.721523626198718425798, 1e-15, |x| x.entropy());
        test_almost(0.0, f64::INFINITY, f64::INFINITY, 1e-15, |x| x.entropy());
    }
    
    #[test]
    fn test_skewness() {
        test_case(0.0, -0.0, 0.0, |x| x.skewness());
        test_case(0.0, 0.0, 0.0, |x| x.skewness());
        test_case(0.0, 0.1, 0.0, |x| x.skewness());
        test_case(4.0, 1.0, 0.0, |x| x.skewness());
        test_case(0.3, 10.0, 0.0, |x| x.skewness());
        test_case(0.0, f64::INFINITY, 0.0, |x| x.skewness());
    }
    
    #[test]
    fn test_mode() {
        // note: std_dev is irrelevant to the mode of a
        // normal distribution
        test_case(-0.0, 0.0, 0.0, |x| x.mode());
        test_case(0.0, 0.0, 0.0, |x| x.mode());
        test_case(0.1, 0.0, 0.1, |x| x.mode());
        test_case(1.0, 0.0, 1.0, |x| x.mode());
        test_case(-10.0, 0.0, -10.0, |x| x.mode());
        test_case(f64::INFINITY, 0.0, f64::INFINITY, |x| x.mode());
    }
    
    #[test]
    fn test_median() {
        // note: std_dev is irrelevant to the median of a
        // normal distribution
        test_optional(-0.0, 0.0, 0.0, |x| x.median());
        test_optional(0.0, 0.0, 0.0, |x| x.median());
        test_optional(0.1, 0.0, 0.1, |x| x.median());
        test_optional(1.0, 0.0, 1.0, |x| x.median());
        test_optional(-0.0, 0.0, -10.0, |x| x.median());
        test_optional(f64::INFINITY, 0.0, f64::INFINITY, |x| x.median());
    }
    
    #[test]
    fn test_min_max() {
        test_case(0.0, 0.0, f64::NEG_INFINITY, |x| x.min());
        test_case(0.0, 0.1, f64::NEG_INFINITY, |x| x.min());
        test_case(-3.0, 10.0, f64::NEG_INFINITY, |x| x.min());
        test_case(0.0, 0.0, f64::INFINITY, |x| x.max());
        test_case(0.0, 0.1, f64::INFINITY, |x| x.max());
        test_case(-3.0, 10.0, f64::INFINITY, |x| x.max());
    }
}