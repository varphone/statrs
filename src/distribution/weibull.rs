use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use consts;
use error::StatsError;
use function::{gamma, stable};
use result::Result;
use super::{Distribution, Univariate, Continuous};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Weibull {
    shape: f64,
    scale: f64,
    scale_pow_shape_inv: f64,
}

impl Weibull {
    pub fn new(shape: f64, scale: f64) -> Result<Weibull> {
        let is_nan = shape.is_nan() || scale.is_nan();
        match (shape, scale, is_nan) {
            (_, _, true) => Err(StatsError::BadParams),
            (_, _, false) if shape <= 0.0 || scale <= 0.0 => Err(StatsError::BadParams),
            (_, _, false) => {
                Ok(Weibull {
                    shape: shape,
                    scale: scale,
                    scale_pow_shape_inv: scale.powf(-shape),
                })
            }
        }
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Sample<f64> for Weibull {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for Weibull {
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution for Weibull {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        let x = r.next_f64();
        self.scale * (-x.ln()).powf(1.0 / self.shape)
    }
}

impl Univariate for Weibull {
    fn mean(&self) -> f64 {
        self.scale * gamma::gamma(1.0 + 1.0 / self.shape)
    }

    fn variance(&self) -> f64 {
        self.scale * self.scale * gamma::gamma(1.0 + 2.0 / self.shape) - self.mean() * self.mean()
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        consts::EULER_MASCHERONI * (1.0 - 1.0 / self.shape) + (self.scale / self.shape).ln() + 1.0
    }

    fn skewness(&self) -> f64 {
        let mu = self.mean();
        let sigma = self.std_dev();
        let sigma2 = sigma * sigma;
        let sigma3 = sigma2 * sigma;
        self.scale * self.scale * self.scale * gamma::gamma(1.0 + 3.0 / self.shape) -
        3.0 * sigma2 * mu - (mu * mu * mu) / sigma3
    }

    fn median(&self) -> f64 {
        self.scale * f64::consts::LN_2.powf(1.0 / self.shape)
    }

    fn cdf(&self, x: f64) -> f64 {
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        -stable::exp_minus_one(x.powf(self.shape) * self.scale_pow_shape_inv)
    }
}

impl Continuous for Weibull {
    fn mode(&self) -> f64 {
        if self.shape <= 1.0 {
            0.0
        } else {
            self.scale * ((self.shape - 1.0) / self.shape).powf(1.0 / self.shape)
        }
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        match (x, self.shape) {
            (0.0, 1.0) => self.shape / self.scale,
            (_, _) if x >= 0.0 => {
                self.shape * (x / self.scale).powf(self.shape - 1.0) *
                (-(x.powf(self.shape)) * self.scale_pow_shape_inv).exp() /
                self.scale
            }
            (_, _) => 0.0,
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        match (x, self.shape) {
            (0.0, 1.0) => self.shape.ln() - self.scale.ln(),
            (_, _) if x >= 0.0 => {
                self.shape.ln() + (self.shape - 1.0) * (x / self.scale).ln() -
                x.powf(self.shape) * self.scale_pow_shape_inv - self.scale.ln()
            }
            (_, _) => f64::NEG_INFINITY, 
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use distribution::{Univariate, Continuous};
    use prec;
    use super::Weibull;

    fn try_create(shape: f64, scale: f64) -> Weibull {
        let n = Weibull::new(shape, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(shape: f64, scale: f64) {
        let n = try_create(shape, scale);
        assert_eq!(shape, n.shape());
        assert_eq!(scale, n.scale());
    }

    fn bad_create_case(shape: f64, scale: f64) {
        let n = Weibull::new(shape, scale);
        assert!(n.is_err());
    }

    fn get_value<F>(shape: f64, scale: f64, eval: F) -> f64
        where F: Fn(Weibull) -> f64
    {
        let n = try_create(shape, scale);
        eval(n)
    }

    fn test_case<F>(shape: f64, scale: f64, expected: f64, eval: F)
        where F: Fn(Weibull) -> f64
    {
        let x = get_value(shape, scale, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(shape: f64, scale: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Weibull) -> f64
    {
        let x = get_value(shape, scale, eval);
        assert!(prec::almost_eq(expected, x, acc));
    }

    #[test]
    fn test_create() {
        create_case(1.0, 0.1);
        create_case(10.0, 1.0);
        create_case(11.0, 10.0);
        create_case(12.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, -1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-1.0, -1.0);
        bad_create_case(0.0, 0.0);
        bad_create_case(0.0, 1.0);
        bad_create_case(1.0, 0.0);
    }

    #[test]
    fn test_mean() {
        test_case(1.0, 0.1, 0.1, |x| x.mean());
        test_case(1.0, 1.0, 1.0, |x| x.mean());
        test_almost(10.0, 10.0, 9.5135076986687318362924871772654021925505786260884, 1e-14, |x| x.mean());
        test_almost(10.0, 1.0, 0.95135076986687318362924871772654021925505786260884, 1e-15, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_almost(1.0, 0.1, 0.01, 1e-16, |x| x.variance());
        test_almost(1.0, 1.0, 1.0, 1e-14, |x| x.variance());
        test_almost(10.0, 10.0, 1.3100455073468309147154581687505295026863354547057, 1e-12, |x| x.variance());
        test_almost(10.0, 1.0, 0.013100455073468309147154581687505295026863354547057, 1e-14, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_almost(1.0, 0.1, 0.1, 1e-15, |x| x.std_dev());
        test_almost(1.0, 1.0, 1.0, 1e-14, |x| x.std_dev());
        test_almost(10.0, 10.0, 1.1445721940300799194124723631014002560036613065794, 1e-12, |x| x.std_dev());
        test_almost(10.0, 1.0, 0.11445721940300799194124723631014002560036613065794, 1e-13, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(1.0, 0.1, -1.302585092994045684018, 1e-15, |x| x.entropy());
        test_case(1.0, 1.0, 1.0, |x| x.entropy());
        test_case(10.0, 10.0, 1.519494098411379574546, |x| x.entropy());
        test_almost(10.0, 1.0, -0.783090994582666109472, 1e-15, |x| x.entropy());
    }
}