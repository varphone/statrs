use std::f64;
use rand::Rng;
use error::StatsError;
use result::Result;
use super::{Distribution, Univariate, Discrete};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DiscreteUniform {
    min: i64,
    max: i64,
}

impl DiscreteUniform {
    pub fn new(min: i64, max: i64) -> Result<DiscreteUniform> {
        if max < min {
            Err(StatsError::BadParams)
        } else {
            Ok(DiscreteUniform {
                min: min,
                max: max,
            })
        }
    }
}

impl Distribution for DiscreteUniform {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        r.gen_range(self.min, self.max + 1) as f64
    }
}

impl Univariate for DiscreteUniform {
    fn mean(&self) -> f64 {
        (self.min + self.max) as f64 / 2.0
    }

    fn variance(&self) -> f64 {
        let diff = (self.max - self.min) as f64;
        ((diff + 1.0) * (diff + 1.0) - 1.0) / 12.0
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        let diff = (self.max - self.min) as f64;
        (diff + 1.0).ln()
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn median(&self) -> f64 {
        (self.min + self.max) as f64 / 2.0
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.min as f64 {
            return 0.0;
        }
        if x >= self.max as f64 {
            return 1.0;
        }

        let lower = self.min as f64;
        let upper = self.max as f64;
        let ans = (x.floor() - lower + 1.0) / (upper - lower + 1.0);
        if x > 1.0 {
            1.0
        } else {
            ans
        }
    }
}

impl Discrete for DiscreteUniform {
    fn mode(&self) -> i64 {
        ((self.min + self.max) as f64 / 2.0).floor() as i64
    }

    fn min(&self) -> i64 {
        self.min
    }

    fn max(&self) -> i64 {
        self.max
    }

    fn pmf(&self, x: i64) -> f64 {
        if x >= self.min && x <= self.max {
            1.0 / (self.max - self.min + 1) as f64
        } else {
            0.0
        }
    }

    fn ln_pmf(&self, x: i64) -> f64 {
        if x >= self.min && x <= self.max {
            -((self.max - self.min + 1) as f64).ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::cmp::PartialEq;
    use std::fmt::Debug;
    use std::f64;
    use distribution::{Univariate, Discrete};
    use super::DiscreteUniform;

    fn try_create(min: i64, max: i64) -> DiscreteUniform {
        let n = DiscreteUniform::new(min, max);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(min: i64, max: i64) {
        let n = try_create(min, max);
        assert_eq!(min, n.min());
        assert_eq!(max, n.max());
    }

    fn bad_create_case(min: i64, max: i64) {
        let n = DiscreteUniform::new(min, max);
        assert!(n.is_err());
    }

    fn test_case<T, F>(min: i64, max: i64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(DiscreteUniform) -> T
    {
        let n = try_create(min, max);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    #[test]
    fn test_create() {
        create_case(-10, 10);
        create_case(0, 4);
        create_case(10, 20);
        create_case(20, 20);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(-1, -2);
        bad_create_case(6, 5);
    }

    #[test]
    fn test_mean() {
        test_case(-10, 10, 0.0, |x| x.mean());
        test_case(0, 4, 2.0, |x| x.mean());
        test_case(10, 20, 15.0, |x| x.mean());
        test_case(20, 20, 20.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(-10, 10, 36.66666666666666666667, |x| x.variance());
        test_case(0, 4, 2.0, |x| x.variance());
        test_case(10, 20, 10.0, |x| x.variance());
        test_case(20, 20, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(-10, 10, (36.66666666666666666667f64).sqrt(), |x| x.std_dev());
        test_case(0, 4, (2.0f64).sqrt(), |x| x.std_dev());
        test_case(10, 20, (10.0f64).sqrt(), |x| x.std_dev());
        test_case(20, 20, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_case(-10, 10, 3.0445224377234229965005979803657054342845752874046093, |x| x.entropy());
        test_case(0, 4, 1.6094379124341003746007593332261876395256013542685181, |x| x.entropy());
        test_case(10, 20, 2.3978952727983705440619435779651292998217068539374197, |x| x.entropy());
        test_case(20, 20, 0.0, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(-10, 10, 0.0, |x| x.skewness());
        test_case(0, 4, 0.0, |x| x.skewness());
        test_case(10, 20, 0.0, |x| x.skewness());
        test_case(20, 20, 0.0, |x| x.skewness());
    }

    #[test]
    fn test_median() {
        test_case(-10, 10, 0.0, |x| x.median());
        test_case(0, 4, 2.0, |x| x.median());
        test_case(10, 20, 15.0, |x| x.median());
        test_case(20, 20, 20.0, |x| x.median());
    }

    #[test]
    fn test_mode() {
        test_case(-10, 10, 0, |x| x.mode());
        test_case(0, 4, 2, |x| x.mode());
        test_case(10, 20, 15, |x| x.mode());
        test_case(20, 20, 20, |x| x.mode());
    }

    #[test]
    fn test_pmf() {
        test_case(-10, 10, 0.04761904761904761904762, |x| x.pmf(-5));
        test_case(-10, 10, 0.04761904761904761904762, |x| x.pmf(1));
        test_case(-10, 10, 0.04761904761904761904762, |x| x.pmf(10));
        test_case(-10, -10, 0.0, |x| x.pmf(0));
        test_case(-10, -10, 1.0, |x| x.pmf(-10));
    }

    #[test]
    fn test_ln_pmf() {
        test_case(-10, 10, -3.0445224377234229965005979803657054342845752874046093, |x| x.ln_pmf(-5));
        test_case(-10, 10, -3.0445224377234229965005979803657054342845752874046093, |x| x.ln_pmf(1));
        test_case(-10, 10, -3.0445224377234229965005979803657054342845752874046093, |x| x.ln_pmf(10));
        test_case(-10, -10, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(-10, -10, 0.0, |x| x.ln_pmf(-10));
    }

    #[test]
    fn test_cdf() {
        test_case(-10, 10, 0.2857142857142857142857, |x| x.cdf(-5.0));
        test_case(-10, 10, 0.5714285714285714285714, |x| x.cdf(1.0));
        test_case(-10, 10, 1.0, |x| x.cdf(10.0));
        test_case(-10, -10, 1.0, |x| x.cdf(0.0));
        test_case(-10, -10, 1.0, |x| x.cdf(-10.0));
        test_case(-10, -10, 0.0, |x| x.cdf(-11.0));
    }
}
