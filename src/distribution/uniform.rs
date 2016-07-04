use std::f64;
use rand::Rng;
use error::StatsError;
use result::Result;
use super::{Distribution, Univariate, CheckedUnivariate, Continuous, CheckedContinuous};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Uniform {
    min: f64,
    max: f64,
}

impl Uniform {
    pub fn new(min: f64, max: f64) -> Result<Uniform> {
        if min > max || min.is_nan() || max.is_nan() {
            Err(StatsError::BadParams)
        } else {
            Ok(Uniform {
                min: min,
                max: max,
            })
        }
    }
}

impl Distribution for Uniform {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        r.gen_range(self.min, self.max + 1.0)
    }
}

impl Univariate for Uniform {
    fn mean(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    fn variance(&self) -> f64 {
        (self.max - self.min) * (self.max - self.min) / 12.0
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        (self.max - self.min).ln()
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn median(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }
        (x - self.min) / (self.max - self.min)
    }
}

impl CheckedUnivariate for Uniform {
    fn checked_mean(&self) -> Result<f64> {
        Ok(self.mean())
    }
    
    fn checked_variance(&self) -> Result<f64> {
        Ok(self.variance())
    }
    
    fn checked_std_dev(&self) -> Result<f64> {
        Ok(self.std_dev())
    }
    
    fn checked_entropy(&self) -> Result<f64> {
        Ok(self.entropy())
    }
    
    fn checked_skewness(&self) -> Result<f64> {
        Ok(self.skewness())
    }
    
    fn checked_median(&self) -> Result<f64> {
        Ok(self.median())
    }
    
    fn checked_cdf(&self, x: f64) -> Result<f64> {
        Ok(self.cdf(x))
    }
}

impl Continuous for Uniform {
    fn mode(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }

    fn pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            1.0 / (self.max - self.min)
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            -(self.max - self.min).ln()
        }
    }
}

impl CheckedContinuous for Uniform {
    fn checked_mode(&self) -> Result<f64> {
        Ok(self.mode())
    }

    fn checked_min(&self) -> Result<f64> {
        Ok(self.min())
    }

    fn checked_max(&self) -> Result<f64> {
        Ok(self.max())
    }

    fn checked_pdf(&self, x: f64) -> Result<f64> {
        Ok(self.pdf(x))
    }

    fn checked_ln_pdf(&self, x: f64) -> Result<f64> {
       Ok(self.ln_pdf(x))
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use distribution::{Univariate, Continuous};
    use prec;
    use super::Uniform;

    fn try_create(min: f64, max: f64) -> Uniform {
        let n = Uniform::new(min, max);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(min: f64, max: f64) {
        let n = try_create(min, max);
        assert_eq!(n.min(), min);
        assert_eq!(n.max(), max);
    }

    fn bad_create_case(min: f64, max: f64) {
        let n = Uniform::new(min, max);
        assert!(n.is_err());
    }

    fn test_case<F>(min: f64, max: f64, expected: f64, eval: F)
        where F: Fn(Uniform) -> f64
    {

        let n = try_create(min, max);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(min: f64, max: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Uniform) -> f64
    {

        let n = try_create(min, max);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));
    }

    #[test]
    fn test_create() {
        create_case(0.0, 0.0);
        create_case(0.0, 0.1);
        create_case(0.0, 1.0);
        create_case(10.0, 10.0);
        create_case(-5.0, 11.0);
        create_case(-5.0, 100.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, 0.0);
    }

    #[test]
    fn test_variance() {
        test_case(-0.0, 2.0, 1.0 / 3.0, |x| x.variance());
        test_case(0.0, 2.0, 1.0 / 3.0, |x| x.variance());
        test_almost(0.1, 4.0, 1.2675, 1e-15, |x| x.variance());
        test_case(10.0, 11.0, 1.0 / 12.0, |x| x.variance());
        test_case(0.0, f64::INFINITY, f64::INFINITY, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(-0.0, 2.0, (1f64 / 3.0).sqrt(), |x| x.std_dev());
        test_case(0.0, 2.0, (1f64 / 3.0).sqrt(), |x| x.std_dev());
        test_almost(0.1, 4.0, (1.2675f64).sqrt(), 1e-15, |x| x.std_dev());
        test_case(10.0, 11.0, (1f64 / 12.0).sqrt(), |x| x.std_dev());
        test_case(0.0, f64::INFINITY, f64::INFINITY, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_case(-0.0, 2.0, 0.6931471805599453094172, |x| x.entropy());
        test_case(0.0, 2.0, 0.6931471805599453094172, |x| x.entropy());
        test_almost(0.1, 4.0, 1.360976553135600743431, 1e-15, |x| x.entropy());
        test_case(1.0, 10.0, 2.19722457733621938279, |x| x.entropy());
        test_case(10.0, 11.0, 0.0, |x| x.entropy());
        test_case(0.0, f64::INFINITY, f64::INFINITY, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_case(-0.0, 2.0, 0.0, |x| x.skewness());
        test_case(0.0, 2.0, 0.0, |x| x.skewness());
        test_case(0.1, 4.0, 0.0, |x| x.skewness());
        test_case(1.0, 10.0, 0.0, |x| x.skewness());
        test_case(10.0, 11.0, 0.0, |x| x.skewness());
        test_case(0.0, f64::INFINITY, 0.0, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(-0.0, 2.0, 1.0, |x| x.mode());
        test_case(0.0, 2.0, 1.0, |x| x.mode());
        test_case(0.1, 4.0, 2.05, |x| x.mode());
        test_case(1.0, 10.0, 5.5, |x| x.mode());
        test_case(10.0, 11.0, 10.5, |x| x.mode());
        test_case(0.0, f64::INFINITY, f64::INFINITY, |x| x.mode());
    }

    #[test]
    fn test_median() {
        test_case(-0.0, 2.0, 1.0, |x| x.median());
        test_case(0.0, 2.0, 1.0, |x| x.median());
        test_case(0.1, 4.0, 2.05, |x| x.median());
        test_case(1.0, 10.0, 5.5, |x| x.median());
        test_case(10.0, 11.0, 10.5, |x| x.median());
        test_case(0.0, f64::INFINITY, f64::INFINITY, |x| x.median());
    }

    #[test]
    fn test_pdf() {
        test_case(0.0, 0.0, 0.0, |x| x.pdf(-5.0));
        test_case(0.0, 0.0, f64::INFINITY, |x| x.pdf(0.0));
        test_case(0.0, 0.0, 0.0, |x| x.pdf(5.0));
        test_case(0.0, 0.1, 0.0, |x| x.pdf(-5.0));
        test_case(0.0, 0.1, 10.0, |x| x.pdf(0.05));
        test_case(0.0, 0.1, 0.0, |x| x.pdf(5.0));
        test_case(0.0, 1.0, 0.0, |x| x.pdf(-5.0));
        test_case(0.0, 1.0, 1.0, |x| x.pdf(0.5));
        test_case(0.0, 0.1, 0.0, |x| x.pdf(5.0));
        test_case(0.0, 10.0, 0.0, |x| x.pdf(-5.0));
        test_case(0.0, 10.0, 0.1, |x| x.pdf(1.0));
        test_case(0.0, 10.0, 0.1, |x| x.pdf(5.0));
        test_case(0.0, 10.0, 0.0, |x| x.pdf(11.0));
        test_case(-5.0, 100.0, 0.0, |x| x.pdf(-10.0));
        test_case(-5.0, 100.0, 0.009523809523809523809524, |x| x.pdf(-5.0));
        test_case(-5.0, 100.0, 0.009523809523809523809524, |x| x.pdf(0.0));
        test_case(-5.0, 100.0, 0.0, |x| x.pdf(101.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(-5.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(10.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(f64::INFINITY));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(0.0, 0.0, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_case(0.0, 0.0, f64::INFINITY, |x| x.ln_pdf(0.0));
        test_case(0.0, 0.0, f64::NEG_INFINITY, |x| x.ln_pdf(5.0));
        test_case(0.0, 0.1, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_almost(0.0, 0.1, 2.302585092994045684018, 1e-15, |x| x.ln_pdf(0.05));
        test_case(0.0, 0.1, f64::NEG_INFINITY, |x| x.ln_pdf(5.0));
        test_case(0.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_case(0.0, 1.0, 0.0, |x| x.ln_pdf(0.5));
        test_case(0.0, 0.1, f64::NEG_INFINITY, |x| x.ln_pdf(5.0));
        test_case(0.0, 10.0, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_case(0.0, 10.0, -2.302585092994045684018, |x| x.ln_pdf(1.0));
        test_case(0.0, 10.0, -2.302585092994045684018, |x| x.ln_pdf(5.0));
        test_case(0.0, 10.0, f64::NEG_INFINITY, |x| x.ln_pdf(11.0));
        test_case(-5.0, 100.0, f64::NEG_INFINITY, |x| x.ln_pdf(-10.0));
        test_case(-5.0, 100.0, -4.653960350157523371101, |x| x.ln_pdf(-5.0));
        test_case(-5.0, 100.0, -4.653960350157523371101, |x| x.ln_pdf(0.0));
        test_case(-5.0, 100.0, f64::NEG_INFINITY, |x| x.ln_pdf(101.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(10.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_cdf() {
        test_case(0.0, 0.0, 0.0, |x| x.cdf(-5.0));
        test_case(0.0, 0.0, 0.0, |x| x.cdf(0.0));
        test_case(0.0, 0.0, 1.0, |x| x.cdf(5.0));
        test_case(0.0, 0.1, 0.0, |x| x.cdf(-5.0));
        test_case(0.0, 0.1, 0.5, |x| x.cdf(0.05));
        test_case(0.0, 0.1, 1.0, |x| x.cdf(5.0));
        test_case(0.0, 1.0, 0.0, |x| x.cdf(-5.0));
        test_case(0.0, 1.0, 0.5, |x| x.cdf(0.5));
        test_case(0.0, 0.1, 1.0, |x| x.cdf(5.0));
        test_case(0.0, 10.0, 0.0, |x| x.cdf(-5.0));
        test_case(0.0, 10.0, 0.1, |x| x.cdf(1.0));
        test_case(0.0, 10.0, 0.5, |x| x.cdf(5.0));
        test_case(0.0, 10.0, 1.0, |x| x.cdf(11.0));
        test_case(-5.0, 100.0, 0.0, |x| x.cdf(-10.0));
        test_case(-5.0, 100.0, 0.0, |x| x.cdf(-5.0));
        test_case(-5.0, 100.0, 0.04761904761904761904762, |x| x.cdf(0.0));
        test_case(-5.0, 100.0, 1.0, |x| x.cdf(101.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.cdf(-5.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.cdf(10.0));
        test_case(0.0, f64::INFINITY, 1.0, |x| x.cdf(f64::INFINITY));
    }
}
