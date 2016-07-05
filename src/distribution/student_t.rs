use std::f64;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use error::StatsError;
use function::{beta, gamma};
use result::Result;
use super::{Distribution, Univariate, Continuous};
use super::normal;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StudentT {
    location: f64,
    scale: f64,
    freedom: f64,
}

impl StudentT {
    pub fn new(location: f64, scale: f64, freedom: f64) -> Result<StudentT> {
        let is_nan = location.is_nan() || scale.is_nan() || freedom.is_nan();
        if is_nan || scale <= 0.0 || freedom <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(StudentT {
                location: location,
                scale: scale,
                freedom: freedom,
            })
        }
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl Sample<f64> for StudentT {
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for StudentT {
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution for StudentT {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        let gamma = super::gamma::sample_unchecked(r, 0.5 * self.freedom, 0.5);
        normal::sample_unchecked(r, self.location, self.scale * (self.freedom / gamma).sqrt())
    }
}

impl Univariate for StudentT {
    fn mean(&self) -> f64 {
        assert!(self.freedom > 1.0,
                format!("{}", StatsError::ArgGt("freedom", 1.0)));
        self.location
    }

    fn variance(&self) -> f64 {
        assert!(self.freedom > 1.0,
                format!("{}", StatsError::ArgGt("freedom", 1.0)));
        if self.freedom == f64::INFINITY {
            self.scale * self.scale
        } else if self.freedom > 2.0 {
            self.freedom * self.scale * self.scale / (self.freedom - 2.0)
        } else {
            f64::INFINITY
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        assert!(self.location == 0.0 && self.scale == 1.0,
                "Cannot calculate entropy for StudentT distribution where location is not 0 and \
                 scale is not 1");

        (self.freedom + 1.0) / 2.0 *
        (gamma::digamma((self.freedom + 1.0) / 2.0) - gamma::digamma(self.freedom / 2.0)) +
        (self.freedom.sqrt() * beta::beta(self.freedom / 2.0, 0.5)).ln()
    }

    fn skewness(&self) -> f64 {
        assert!(self.freedom > 3.0,
                format!("{}", StatsError::ArgGt("freedom", 3.0)));
        0.0
    }

    fn median(&self) -> f64 {
        self.location
    }

    fn cdf(&self, x: f64) -> f64 {
        if self.freedom == f64::INFINITY {
            normal::cdf_unchecked(x, self.location, self.scale)
        } else {
            let k = (x - self.location) / self.scale;
            let h = self.freedom / (self.freedom + k * k);
            let ib = 0.5 * beta::beta_reg(self.freedom / 2.0, 0.5, h);
            if x <= self.location {
                ib
            } else {
                1.0 - ib
            }
        }
    }
}

impl Continuous for StudentT {
    fn mode(&self) -> f64 {
        self.location
    }

    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        if self.freedom >= 1e8 {
            normal::pdf_unchecked(x, self.location, self.scale)
        } else {
            let d = (x - self.location) / self.scale;
            (gamma::ln_gamma((self.freedom + 1.0) / 2.0) - gamma::ln_gamma(self.freedom / 2.0))
                .exp() *
            (1.0 + d * d / self.freedom).powf(-0.5 * (self.freedom + 1.0)) /
            (self.freedom * f64::consts::PI).sqrt() / self.scale
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if self.freedom >= 1e8 {
            normal::ln_pdf_unchecked(x, self.location, self.scale)
        } else {
            let d = (x - self.location) / self.scale;
            gamma::ln_gamma((self.freedom + 1.0) / 2.0) -
            0.5 * ((self.freedom + 1.0) * (1.0 + d * d / self.freedom).ln()) -
            gamma::ln_gamma(self.freedom / 2.0) -
            0.5 * (self.freedom * f64::consts::PI).ln() - self.scale.ln()
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use distribution::{Univariate, Continuous};
    use prec;
    use super::StudentT;

    fn try_create(location: f64, scale: f64, freedom: f64) -> StudentT {
        let n = StudentT::new(location, scale, freedom);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(location: f64, scale: f64, freedom: f64) {
        let n = try_create(location, scale, freedom);
        assert_eq!(n.location(), location);
        assert_eq!(n.scale(), scale);
        assert_eq!(n.freedom(), freedom);
    }

    fn bad_create_case(location: f64, scale: f64, freedom: f64) {
        let n = StudentT::new(location, scale, freedom);
        assert!(n.is_err());
    }

    fn test_case<F>(location: f64, scale: f64, freedom: f64, expected: f64, eval: F)
        where F: Fn(StudentT) -> f64
    {

        let n = try_create(location, scale, freedom);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(location: f64, scale: f64, freedom: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(StudentT) -> f64
    {

        let n = try_create(location, scale, freedom);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));
    }

    #[test]
    fn test_create() {
        create_case(0.0, 0.1, 1.0);
        create_case(0.0, 1.0, 1.0);
        create_case(-5.0, 1.0, 3.0);
        create_case(10.0, 10.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0, 1.0);
        bad_create_case(0.0, f64::NAN, 1.0);
        bad_create_case(0.0, 1.0, f64::NAN);
        bad_create_case(0.0, -10.0, 1.0);
        bad_create_case(0.0, 10.0, -1.0);
    }
}
