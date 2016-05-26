use std::f64;
use rand::Rng;
use error::StatsError;
use function::gamma;
use result::Result;
use super::{Distribution, Univariate, Continuous};
use super::normal;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Chi {
    freedom: f64,
}

impl Chi {
    pub fn new(freedom: f64) -> Result<Chi> {
        if freedom.is_nan() || freedom <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Chi { freedom: freedom })
        }
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl Distribution for Chi {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        (0..self.freedom as i64)
            .fold(0.0,
                  |acc, _| acc + normal::sample_unchecked(r, 0.0, 1.0).powf(2.0))
            .sqrt()
    }
}

impl Univariate for Chi {
    fn mean(&self) -> f64 {
        f64::consts::SQRT_2 * gamma::gamma((self.freedom + 1.0) / 2.0) /
        gamma::gamma(self.freedom / 2.0)
    }

    fn variance(&self) -> f64 {
        self.freedom - self.mean() * self.mean()
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        gamma::ln_gamma(self.freedom / 2.0) +
        (self.freedom - (2.0f64).ln() -
         (self.freedom - 1.0) * gamma::digamma(self.freedom / 2.0).unwrap()) / 2.0
    }

    fn skewness(&self) -> f64 {
        let sigma = self.std_dev();
        self.mean() * (1.0 - 2.0 * sigma * sigma) / (sigma * sigma * sigma)
    }

    fn median(&self) -> f64 {
        unimplemented!()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x == f64::INFINITY || self.freedom == f64::INFINITY {
            1.0
        } else {
            gamma::gamma_lr(self.freedom / 2.0, x * x / 2.0).unwrap()
        }
    }
}

impl Continuous for Chi {
    fn mode(&self) -> f64 {
        if self.freedom < 1.0 {
            panic!("Cannot calculate Chi distribution mode for freedom < 1");
        }
        (self.freedom - 1.0).sqrt()
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        match (self.freedom, x) {
            (f64::INFINITY, _) | (_, f64::INFINITY) | (_, 0.0) => 0.0,
            (_, _) if self.freedom > 160.0 => self.ln_pdf(x),
            (_, _) => {
                (2.0f64).powf(1.0 - self.freedom / 2.0) * x.powf(self.freedom - 1.0) *
                (-x * x / 2.0).exp() / gamma::gamma(self.freedom / 2.0)
            }
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        match (self.freedom, x) {
            (f64::INFINITY, _) | (_, f64::INFINITY) | (_, 0.0) => f64::NEG_INFINITY,
            (_, _) => {
                (1.0 - self.freedom / 2.0) * (2.0f64).ln() + ((self.freedom - 1.0) * x.ln()) -
                x * x / 2.0 - gamma::ln_gamma(self.freedom / 2.0)
            }
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use distribution::{Univariate, Continuous};
    use prec;
    use super::Chi;
    
    fn try_create(freedom: f64) -> Chi {
        let n = Chi::new(freedom);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(freedom: f64) {
        let n = try_create(freedom);
        assert_eq!(freedom, n.freedom());
    }

    fn bad_create_case(freedom: f64) {
        let n = Chi::new(freedom);
        assert!(n.is_err());
    }

    fn test_case<F>(freedom: f64, expected: f64, eval: F)
        where F: Fn(Chi) -> f64
    {
        let n = try_create(freedom);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(freedom: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Chi) -> f64
    {
        let n = try_create(freedom);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));
    }
    
    fn test_is_nan<F>(freedom: f64, eval: F)
        where F : Fn(Chi) -> f64 
    {
        let n = try_create(freedom);
        let x = eval(n);
        assert!(x.is_nan());
    }
    
    #[test]
    fn test_create() {
        create_case(1.0);
        create_case(3.0);
        create_case(f64::INFINITY);
    }
    
    #[test]
    fn test_bad_create() {
        bad_create_case(0.0);
        bad_create_case(-1.0);
        bad_create_case(-100.0);
        bad_create_case(f64::NEG_INFINITY);
        bad_create_case(f64::NAN);
    }
    
    #[test]
    fn test_mean() {
        test_almost(1.0, 0.7978845608028653558799, 1e-15, |x| x.mean());
        test_almost(2.0, 1.25331413731550025121, 1e-14, |x| x.mean());
        test_almost(2.5, 1.43396639245837498609, 1e-14, |x| x.mean());
        test_almost(5.0, 2.12769216214097428235, 1e-14, |x| x.mean());
        test_is_nan(f64::INFINITY, |x| x.mean());
    }
    
    #[test]
    fn test_variance() {
        test_almost(1.0, 0.3633802276324186569245, 1e-15, |x| x.variance());
        test_almost(2.0, 0.42920367320510338077, 1e-14, |x| x.variance());
        test_almost(2.5, 0.44374038529991368581, 1e-13, |x| x.variance());
        test_almost(3.0, 0.4535209105296746277, 1e-14, |x| x.variance());
        test_is_nan(f64::INFINITY, |x| x.variance());
    }
    
    #[test]
    fn test_std_dev() {
        test_almost(1.0, 0.6028102749890869742759, 1e-15, |x| x.std_dev());
        test_almost(2.0, 0.65513637756203355309, 1e-14, |x| x.std_dev());
        test_almost(2.5, 0.66613841301933165564, 1e-14, |x| x.std_dev());
        test_almost(3.0, 0.67343961164285148374, 1e-14, |x| x.std_dev());
        test_is_nan(f64::INFINITY, |x| x.std_dev());
    }
    
    #[test]
    fn test_entropy() {
        test_almost(1.0, 0.7257913526447274323631, 1e-15, |x| x.entropy());
        test_almost(2.0, 0.9420342421707937755946, 1e-15, |x| x.entropy());
        test_almost(2.5, 0.97574472333041323989, 1e-14, |x| x.entropy());
        test_almost(3.0, 0.99615419810620560239, 1e-14, |x| x.entropy());
        test_is_nan(f64::INFINITY, |x| x.entropy());
    }
    
    #[test]
    fn test_skewness() {
        test_almost(1.0, 0.995271746431156042444, 1e-14, |x| x.skewness());
        test_almost(2.0, 0.6311106578189371382, 1e-13, |x| x.skewness());
        test_almost(2.5, 0.5458487096285153216, 1e-12, |x| x.skewness());
        test_almost(3.0, 0.485692828049590809, 1e-12, |x| x.skewness());
        test_is_nan(f64::INFINITY, |x| x.skewness());
    }
    
    #[test]
    #[should_panic]
    fn test_median() {
        // should always panic
        test_case(1.0, 0.0, |x| x.median());
        test_case(2.0, 0.0, |x| x.median());
        test_case(2.5, 0.0, |x| x.median());
        test_case(3.0, 0.0, |x| x.median());
        test_case(f64::INFINITY, 0.0, |x| x.median());
    }
    
    #[test]
    fn test_mode() {
        test_case(1.0, 0.0, |x| x.mode());
        test_case(2.0, 1.0, |x| x.mode());
        test_case(2.5, 1.224744871391589049099, |x| x.mode());
        test_case(3.0, f64::consts::SQRT_2, |x| x.mode());
        test_case(f64::INFINITY, f64::INFINITY, |x| x.mode());
    }
    
    #[test]
    fn test_min_max() {
        test_case(1.0, 0.0, |x| x.min());
        test_case(2.0, 0.0, |x| x.min());
        test_case(2.5, 0.0, |x| x.min());
        test_case(3.0, 0.0, |x| x.min());
        test_case(f64::INFINITY, 0.0, |x| x.min());
        test_case(1.0, f64::INFINITY, |x| x.max());
        test_case(2.0, f64::INFINITY, |x| x.max());
        test_case(2.5, f64::INFINITY, |x| x.max());
        test_case(3.0, f64::INFINITY, |x| x.max());
        test_case(f64::INFINITY, f64::INFINITY, |x| x.max());
    }
}