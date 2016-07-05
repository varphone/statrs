use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use error::StatsError;
use function::{beta, factorial};
use result::Result;
use super::{Distribution, Univariate, Discrete};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Binomial {
    p: f64,
    n: i64,
}

impl Binomial {
    pub fn new(p: f64, n: i64) -> Result<Binomial> {
        if p.is_nan() || p < 0.0 || p > 1.0 || n < 0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Binomial { p: p, n: n })
        }
    }

    pub fn p(&self) -> f64 {
        self.p
    }

    pub fn n(&self) -> i64 {
        self.n
    }
}

impl Sample<f64> for Binomial {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Binomial {
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution for Binomial {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        (0..self.n).fold(0.0, |acc, _| {
            let n = r.next_f64();
            if n < self.p {
                acc + 1.0
            } else {
                acc
            }
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
                (0..self.n + 1).fold(0.0, |acc, x| {
                    let p = self.pmf(x);
                    acc - p * p.ln()
                })
            }
        }
    }

    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * self.p) / (self.n as f64 * self.p * (1.0 - self.p)).sqrt()
    }

    fn median(&self) -> f64 {
        (self.p * self.n as f64).floor()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x >= self.n as f64 {
            1.0
        } else {
            let k = x.floor();
            beta::beta_reg(self.n as f64 - k, k + 1.0, 1.0 - self.p)
        }
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
        if x > self.n || x < 0 {
            0.0
        } else {
            match self.p {
                0.0 if x == 0 => 1.0,
                0.0 => 0.0,
                1.0 if x == self.n => 1.0,
                1.0 => 0.0,
                _ => {
                    (factorial::ln_binomial(self.n as u64, x as u64) + x as f64 * self.p.ln() +
                     (self.n - x) as f64 * (1.0 - self.p).ln())
                        .exp()
                }
            }
        }
    }

    fn ln_pmf(&self, x: i64) -> f64 {
        if x > self.n || x < 0 {
            f64::NEG_INFINITY
        } else {
            match self.p {
                0.0 if x == 0 => 0.0,
                0.0 => f64::NEG_INFINITY,
                1.0 if x == self.n => 0.0,
                1.0 => f64::NEG_INFINITY,
                _ => {
                    factorial::ln_binomial(self.n as u64, x as u64) + x as f64 * self.p.ln() +
                    (self.n - x) as f64 * (1.0 - self.p).ln()
                }
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
    use distribution::{Univariate, Discrete};
    use prec;
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

    fn get_value<T, F>(p: f64, n: i64, eval: F) -> T
        where T: PartialEq + Debug,
              F: Fn(Binomial) -> T
    {
        let n = try_create(p, n);
        eval(n)
    }

    fn test_case<T, F>(p: f64, n: i64, expected: T, eval: F)
        where T: PartialEq + Debug,
              F: Fn(Binomial) -> T
    {
        let x = get_value(p, n, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(p: f64, n: i64, expected: f64, acc: f64, eval: F)
        where F: Fn(Binomial) -> f64
    {
        let x = get_value(p, n, eval);
        assert!(prec::almost_eq(expected, x, acc));
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

    #[test]
    fn test_mean() {
        test_case(0.0, 4, 0.0, |x| x.mean());
        test_almost(0.3, 3, 0.9, 1e-15, |x| x.mean());
        test_case(1.0, 2, 2.0, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(0.0, 4, 0.0, |x| x.variance());
        test_case(0.3, 3, 0.63, |x| x.variance());
        test_case(1.0, 2, 0.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(0.0, 4, 0.0, |x| x.std_dev());
        test_case(0.3, 3, 0.7937253933193771771505, |x| x.std_dev());
        test_case(1.0, 2, 0.0, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_case(0.0, 4, 0.0, |x| x.entropy());
        test_almost(0.3, 3, 1.1404671643037712668976423399228972051669206536461, 1e-15, |x| x.entropy());
        test_case(1.0, 2, 0.0, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(0.0, 4, f64::INFINITY, |x| x.skewness());
        test_case(0.3, 3, 0.503952630678969636286, |x| x.skewness());
        test_case(1.0, 2, f64::NEG_INFINITY, |x| x.skewness());
    }

    #[test]
    fn test_median() {
        test_case(0.0, 4, 0.0, |x| x.median());
        test_case(0.3, 3, 0.0, |x| x.median());
        test_case(1.0, 2, 2.0, |x| x.median());
    }

    #[test]
    fn test_mode() {
        test_case(0.0, 4, 0, |x| x.mode());
        test_case(0.3, 3, 1, |x| x.mode());
        test_case(1.0, 2, 2, |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(0.3, 10, 0, |x| x.min());
        test_case(0.3, 10, 10, |x| x.max());
    }

    #[test]
    fn test_pmf() {
        test_case(0.0, 1, 1.0, |x| x.pmf(0));
        test_case(0.0, 1, 0.0, |x| x.pmf(1));
        test_case(0.0, 3, 1.0, |x| x.pmf(0));
        test_case(0.0, 3, 0.0, |x| x.pmf(1));
        test_case(0.0, 3, 0.0, |x| x.pmf(3));
        test_case(0.0, 10, 1.0, |x| x.pmf(0));
        test_case(0.0, 10, 0.0, |x| x.pmf(1));
        test_case(0.0, 10, 0.0, |x| x.pmf(10));
        test_case(0.3, 1, 0.69999999999999995559107901499373838305473327636719, |x| x.pmf(0));
        test_case(0.3, 1, 0.2999999999999999888977697537484345957636833190918, |x| x.pmf(1));
        test_case(0.3, 3, 0.34299999999999993471888615204079956461021032657166, |x| x.pmf(0));
        test_almost(0.3, 3, 0.44099999999999992772448109690231306411849135972008, 1e-15, |x| x.pmf(1));
        test_almost(0.3, 3, 0.026999999999999997002397833512077451789759292859569, 1e-16, |x| x.pmf(3));
        test_almost(0.3, 10, 0.02824752489999998207939855277004937778546385011091, 1e-17, |x| x.pmf(0));
        test_almost(0.3, 10, 0.12106082099999992639752977030555903089040470780077, 1e-15, |x| x.pmf(1));
        test_almost(0.3, 10, 0.0000059048999999999978147480206303047454017251032868501, 1e-20, |x| x.pmf(10));
        test_case(1.0, 1, 0.0, |x| x.pmf(0));
        test_case(1.0, 1, 1.0, |x| x.pmf(1));
        test_case(1.0, 3, 0.0, |x| x.pmf(0));
        test_case(1.0, 3, 0.0, |x| x.pmf(1));
        test_case(1.0, 3, 1.0, |x| x.pmf(3));
        test_case(1.0, 10, 0.0, |x| x.pmf(0));
        test_case(1.0, 10, 0.0, |x| x.pmf(1));
        test_case(1.0, 10, 1.0, |x| x.pmf(10));
    }

    #[test]
    fn test_ln_pmf() {
        test_case(0.0, 1, 0.0, |x| x.ln_pmf(0));
        test_case(0.0, 1, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.0, 3, 0.0, |x| x.ln_pmf(0));
        test_case(0.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(3));
        test_case(0.0, 10, 0.0, |x| x.ln_pmf(0));
        test_case(0.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(0.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(10));
        test_case(0.3, 1, -0.3566749439387324423539544041072745145718090708995, |x| x.ln_pmf(0));
        test_case(0.3, 1, -1.2039728043259360296301803719337238685164245381839, |x| x.ln_pmf(1));
        test_case(0.3, 3, -1.0700248318161973270618632123218235437154272126985, |x| x.ln_pmf(0));
        test_almost(0.3, 3, -0.81871040353529122294284394322574719301255212216016, 1e-15, |x| x.ln_pmf(1));
        test_almost(0.3, 3, -3.6119184129778080888905411158011716055492736145517, 1e-15, |x| x.ln_pmf(3));
        test_case(0.3, 10, -3.566749439387324423539544041072745145718090708995, |x| x.ln_pmf(0));
        test_almost(0.3, 10, -2.1114622067804823267977785542148302920616046876506, 1e-14, |x| x.ln_pmf(1));
        test_case(0.3, 10, -12.039728043259360296301803719337238685164245381839, |x| x.ln_pmf(10));
        test_case(1.0, 1, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(1.0, 1, 0.0, |x| x.ln_pmf(1));
        test_case(1.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(1.0, 3, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(1.0, 3, 0.0, |x| x.ln_pmf(3));
        test_case(1.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(0));
        test_case(1.0, 10, f64::NEG_INFINITY, |x| x.ln_pmf(1));
        test_case(1.0, 10, 0.0, |x| x.ln_pmf(10));
    }

    #[test]
    fn test_cdf() {
        test_case(0.0, 1, 1.0, |x| x.cdf(0.0));
        test_case(0.0, 1, 1.0, |x| x.cdf(1.0));
        test_case(0.0, 3, 1.0, |x| x.cdf(0.0));
        test_case(0.0, 3, 1.0, |x| x.cdf(1.0));
        test_case(0.0, 3, 1.0, |x| x.cdf(3.0));
        test_case(0.0, 10, 1.0, |x| x.cdf(0.0));
        test_case(0.0, 10, 1.0, |x| x.cdf(1.0));
        test_case(0.0, 10, 1.0, |x| x.cdf(10.0));
        test_almost(0.3, 1, 0.7, 1e-15, |x| x.cdf(0.0));
        test_case(0.3, 1, 1.0, |x| x.cdf(1.0));
        test_almost(0.3, 3, 0.343, 1e-14, |x| x.cdf(0.0));
        test_almost(0.3, 3, 0.784, 1e-15, |x| x.cdf(1.0));
        test_case(0.3, 3, 1.0, |x| x.cdf(3.0));
        test_almost(0.3, 10, 0.0282475249, 1e-16, |x| x.cdf(0.0));
        test_almost(0.3, 10, 0.1493083459, 1e-14, |x| x.cdf(1.0));
        test_case(0.3, 10, 1.0, |x| x.cdf(10.0));
        test_case(1.0, 1, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 1, 1.0, |x| x.cdf(1.0));
        test_case(1.0, 3, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 3, 0.0, |x| x.cdf(1.0));
        test_case(1.0, 3, 1.0, |x| x.cdf(3.0));
        test_case(1.0, 10, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 10, 0.0, |x| x.cdf(1.0));
        test_case(1.0, 10, 1.0, |x| x.cdf(10.0));
    }
}
