use std::f64;
use rand::Rng;
use error::StatsError;
use result::Result;
use super::{Distribution, Univariate, Continuous};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Exponential {
    rate: f64,
}

impl Exponential {
    pub fn new(rate: f64) -> Result<Exponential> {
        if rate.is_nan() || rate <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Exponential { rate: rate })
        }
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Distribution for Exponential {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        let mut x = r.next_f64();
        while x == 0.0 {
            x = r.next_f64();
        }
        -x.ln() / self.rate
    }
}

impl Univariate for Exponential {
    fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }

    fn std_dev(&self) -> f64 {
        1.0 / self.rate
    }

    fn entropy(&self) -> f64 {
        1.0 - self.rate.ln()
    }

    fn skewness(&self) -> f64 {
        2.0
    }

    fn median(&self) -> f64 {
        f64::consts::LN_2 / self.rate
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.rate * x).exp()
        }
    }
}

impl Continuous for Exponential {
    fn mode(&self) -> f64 {
        0.0
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.rate.ln() - self.rate * x
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use distribution::{Univariate, Continuous};
    use prec;
    use super::Exponential;
    
    fn try_create(rate: f64) -> Exponential {
        let n = Exponential::new(rate);
        assert!(n.is_ok());
        n.unwrap()
    }
    
    fn create_case(rate: f64) {
        let n = try_create(rate);
        assert_eq!(rate, n.rate());
    }
    
    fn bad_create_case(rate: f64) {
        let n = Exponential::new(rate);
        assert!(n.is_err());
    }

    fn test_case<F>(rate: f64, expected: f64, eval: F)
        where F: Fn(Exponential) -> f64
    {
        let n = try_create(rate);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(rate: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Exponential) -> f64
    {
        let n = try_create(rate);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));
    }
    
    #[test]
    fn test_create() {
        create_case(0.1);
        create_case(1.0);
        create_case(10.0);
    }
    
    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
        bad_create_case(0.0);
        bad_create_case(-1.0);
        bad_create_case(-10.0);
    }
    
    #[test]
    fn test_mean() {
        test_case(0.1, 10.0, |x| x.mean());
        test_case(1.0, 1.0, |x| x.mean());
        test_case(10.0, 0.1, |x| x.mean());
    }
    
    #[test]
    fn test_variance() {
        test_almost(0.1, 100.0, 1e-13, |x| x.variance());
        test_case(1.0, 1.0, |x| x.variance());
        test_case(10.0, 0.01, |x| x.variance());
    }
    
    #[test]
    fn test_std_dev() {
        test_case(0.1, 10.0, |x| x.std_dev());
        test_case(1.0, 1.0, |x| x.std_dev());
        test_case(10.0, 0.1, |x| x.std_dev());
    }
    
    #[test]
    fn test_entropy() {
        test_almost(0.1, 3.302585092994045684018, 1e-15, |x| x.entropy());
        test_case(1.0, 1.0, |x| x.entropy());
        test_almost(10.0, -1.302585092994045684018, 1e-15, |x| x.entropy());
    }
    
    #[test]
    fn test_skewness() {
        test_case(0.1, 2.0, |x| x.skewness());
        test_case(1.0, 2.0, |x| x.skewness());
        test_case(10.0, 2.0, |x| x.skewness());
    }
    
    #[test]
    fn test_median() {
        test_almost(0.1, 6.931471805599453094172, 1e-15, |x| x.median());
        test_case(1.0, f64::consts::LN_2, |x| x.median());
        test_case(10.0, 0.06931471805599453094172, |x| x.median());
    }
    
    #[test]
    fn test_mode() {
        test_case(0.1, 0.0, |x| x.mode());
        test_case(1.0, 0.0, |x| x.mode());
        test_case(10.0, 0.0, |x| x.mode());
    }
    
    #[test]
    fn test_min_max() {
        test_case(0.1, 0.0, |x| x.min());
        test_case(1.0, 0.0, |x| x.min());
        test_case(10.0, 0.0, |x| x.min());
        test_case(0.1, f64::INFINITY, |x| x.max());
        test_case(1.0, f64::INFINITY, |x| x.max());
        test_case(10.0, f64::INFINITY, |x| x.max());
    }
}