use std::f64;
use std::i64;
use rand::Rng;
use error::StatsError;
use function::{factorial, gamma};
use result::Result;
use super::{Distribution, Univariate, Discrete};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Result<Poisson> {
        if lambda.is_nan() || lambda <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Poisson { lambda: lambda })
        }
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution for Poisson {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.lambda)
    }
}

impl Univariate for Poisson {
    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        self.lambda
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        0.5 * (2.0 * f64::consts::PI * f64::consts::E * self.lambda).ln() -
        1.0 / (12.0 * self.lambda) - 1.0 / (24.0 * self.lambda * self.lambda) -
        19.0 / (360.0 * self.lambda * self.lambda * self.lambda)
    }

    fn skewness(&self) -> f64 {
        1.0 / self.lambda.sqrt()
    }

    fn median(&self) -> f64 {
        (self.lambda + 1.0 / 3.0 - 0.02 / self.lambda).floor()
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 - gamma::gamma_lr(x + 1.0, self.lambda).unwrap()
    }
}

impl Discrete for Poisson {
    fn mode(&self) -> i64 {
        self.lambda.floor() as i64
    }

    fn min(&self) -> i64 {
        0
    }

    fn max(&self) -> i64 {
        i64::MAX
    }

    fn pmf(&self, x: i64) -> f64 {
        if x < 0 {
            panic!("{}", StatsError::ArgNotNegative("x"));
        }
        (-self.lambda + x as f64 * self.lambda.ln() - factorial::ln_factorial(x as u64)).exp()
    }

    fn ln_pmf(&self, x: i64) -> f64 {
        if x < 0 {
            panic!("{}", StatsError::ArgNotNegative("x"));
        }
        -self.lambda + x as f64 * self.lambda.ln() - factorial::ln_factorial(x as u64)
    }
}

/// Generates one sample from the Poisson distribution either by
/// Knuth's method if lambda < 30.0 or Rejection method PA by
/// A. C. Atkinson from the Journal of the Royal Statistical Society
/// Series C (Applied Statistics) Vol. 28 No. 1. (1979) pp. 29 - 35
/// otherwise
fn sample_unchecked<R: Rng>(r: &mut R, lambda: f64) -> f64 {
    if lambda < 30.0 {
        let limit = (-lambda).exp();
        let mut count = 0.0;
        let mut product = r.next_f64();
        while product >= limit {
            count += 1.0;
            product *= r.next_f64();
        }
        count
    } else {
        let c = 0.767 - 3.36 / lambda;
        let beta = f64::consts::PI / (3.0 * lambda).sqrt();
        let alpha = beta * lambda;
        let k = c.ln() - lambda - beta.ln();

        loop {
            let u = r.next_f64();
            let x = (alpha - ((1.0 - u) / u).ln()) / beta;
            let n = (x + 0.5).floor();
            if n < 0.0 {
                continue;
            }

            let v = r.next_f64();
            let y = alpha - beta * x;
            let temp = 1.0 + y.exp();
            let lhs = y + (v / (temp * temp)).ln();
            let rhs = k + n * lambda.ln() - factorial::ln_factorial(n as u64);
            if lhs <= rhs {
                return n;
            }
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::cmp::PartialEq;
    use std::fmt::Debug;
    use std::f64;
    use std::i64;
    use distribution::{Univariate, Discrete};
    use prec;
    use super::Poisson;
    
    fn try_create(lambda: f64) -> Poisson {
        let n = Poisson::new(lambda);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(lambda: f64) {
        let n = try_create(lambda);
        assert_eq!(lambda, n.lambda());
    }

    fn bad_create_case(lambda: f64) {
        let n = Poisson::new(lambda);
        assert!(n.is_err());
    }

    fn test_case<T, F>(lambda: f64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Poisson) -> T
    {

        let n = try_create(lambda);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(lambda: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Poisson) -> f64
    {

        let n = try_create(lambda);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));
    }
    
    #[test]
    fn test_create() {
        create_case(1.5);
        create_case(5.4);
        create_case(10.8);
    }
    
    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN);
        bad_create_case(-1.5);
        bad_create_case(0.0);
    }
    
    #[test]
    fn test_mean() {
        test_case(1.5, 1.5, |x| x.mean());
        test_case(5.4, 5.4, |x| x.mean());
        test_case(10.8, 10.8, |x| x.mean());
    }
    
    #[test]
    fn test_variance() {
        test_case(1.5, 1.5, |x| x.variance());
        test_case(5.4, 5.4, |x| x.variance());
        test_case(10.8, 10.8, |x| x.variance());
    }
    
    #[test]
    fn test_std_dev() {
        test_case(1.5, (1.5f64).sqrt(), |x| x.std_dev());
        test_case(5.4, (5.4f64).sqrt(), |x| x.std_dev());
        test_case(10.8, (10.8f64).sqrt(), |x| x.std_dev());
    }
    
    #[test]
    fn test_entropy() {
        test_almost(1.5, 1.531959153102376331946, 1e-15, |x| x.entropy());
        test_almost(5.4, 2.244941839577643504608, 1e-15, |x| x.entropy());
        test_case(10.8, 2.600596429676975222694, |x| x.entropy());
    }
    
    #[test]
    fn test_skewness() {
        test_almost(1.5, 0.8164965809277260327324, 1e-15, |x| x.skewness());
        test_almost(5.4, 0.4303314829119352094644, 1e-16, |x| x.skewness());
        test_almost(10.8, 0.3042903097250922852539, 1e-16, |x| x.skewness());
    }
    
    #[test]
    fn test_median() {
        test_case(1.5, 1.0, |x| x.median());
        test_case(5.4, 5.0, |x| x.median());
        test_case(10.8, 11.0, |x| x.median());
    }
    
    #[test]
    fn test_mode() {
        test_case(1.5, 1, |x| x.mode());
        test_case(5.4, 5, |x| x.mode());
        test_case(10.8, 10, |x| x.mode());
    }
    
    #[test]
    fn test_min_max() {
        test_case(1.5, 0, |x| x.min());
        test_case(5.4, 0, |x| x.min());
        test_case(10.8, 0, |x| x.min());
        test_case(1.5, i64::MAX, |x| x.max());
        test_case(5.4, i64::MAX, |x| x.max());
        test_case(10.8, i64::MAX, |x| x.max());
    }
}
