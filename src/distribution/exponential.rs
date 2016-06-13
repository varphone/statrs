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
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        1.0 - (-self.rate * x).exp()
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
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
        self.rate * (-self.rate * x).exp()
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        assert!(x >= 0.0, format!("{}", StatsError::ArgNotNegative("x")));
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

    fn get_value<F>(rate: f64, eval: F) -> f64
        where F: Fn(Exponential) -> f64
    {
        let n = try_create(rate);
        eval(n)
    }

    fn test_case<F>(rate: f64, expected: f64, eval: F)
        where F: Fn(Exponential) -> f64
    {
        let x = get_value(rate, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(rate: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Exponential) -> f64
    {
        let x = get_value(rate, eval);
        assert!(prec::almost_eq(expected, x, acc));
    }

    fn test_is_nan<F>(rate: f64, eval: F)
        where F : Fn(Exponential) -> f64
    {
        let x = get_value(rate, eval);
        assert!(x.is_nan());
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

    #[test]
    fn test_pdf() {
        test_case(0.1, 0.1, |x| x.pdf(0.0));
        test_case(1.0, 1.0, |x| x.pdf(0.0));
        test_case(10.0, 10.0, |x| x.pdf(0.0));
        test_is_nan(f64::INFINITY, |x| x.pdf(0.0));
        test_case(0.1, 0.09900498337491680535739, |x| x.pdf(0.1));
        test_almost(1.0, 0.9048374180359595731642, 1e-15, |x| x.pdf(0.1));
        test_case(10.0, 3.678794411714423215955, |x| x.pdf(0.1));
        test_is_nan(f64::INFINITY, |x| x.pdf(0.1));
        test_case(0.1, 0.09048374180359595731642, |x| x.pdf(1.0));
        test_case(1.0, 0.3678794411714423215955, |x| x.pdf(1.0));
        test_almost(10.0, 4.539992976248485153559e-4, 1e-19, |x| x.pdf(1.0));
        test_is_nan(f64::INFINITY, |x| x.pdf(1.0));
        test_case(0.1, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(1.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(10.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, |x| x.pdf(f64::INFINITY));
    }

    #[test]
    #[should_panic]
    fn test_neg_pdf() {
        get_value(0.1, |x| x.pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_almost(0.1, -2.302585092994045684018, 1e-15, |x| x.ln_pdf(0.0));
        test_case(1.0, 0.0, |x| x.ln_pdf(0.0));
        test_case(10.0, 2.302585092994045684018, |x| x.ln_pdf(0.0));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(0.0));
        test_almost(0.1, -2.312585092994045684018, 1e-15, |x| x.ln_pdf(0.1));
        test_case(1.0, -0.1, |x| x.ln_pdf(0.1));
        test_almost(10.0, 1.302585092994045684018, 1e-15, |x| x.ln_pdf(0.1));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(0.1));
        test_case(0.1, -2.402585092994045684018, |x| x.ln_pdf(1.0));
        test_case(1.0, -1.0, |x| x.ln_pdf(1.0));
        test_case(10.0, -7.697414907005954315982, |x| x.ln_pdf(1.0));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(1.0));
        test_case(0.1, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(1.0, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(10.0, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_is_nan(f64::INFINITY, |x| x.ln_pdf(f64::INFINITY));
    }

    #[test]
    #[should_panic]
    fn test_neg_ln_pdf() {
        get_value(0.1, |x| x.ln_pdf(-1.0));
    }

    #[test]
    fn test_cdf() {
        test_case(0.1, 0.0, |x| x.cdf(0.0));
        test_case(1.0, 0.0, |x| x.cdf(0.0));
        test_case(10.0, 0.0, |x| x.cdf(0.0));
        test_is_nan(f64::INFINITY, |x| x.cdf(0.0));
        test_almost(0.1, 0.009950166250831946426094, 1e-16, |x| x.cdf(0.1));
        test_almost(1.0, 0.0951625819640404268358, 1e-16, |x| x.cdf(0.1));
        test_case(10.0, 0.6321205588285576784045, |x| x.cdf(0.1));
        test_case(f64::INFINITY, 1.0, |x| x.cdf(0.1));
        test_almost(0.1, 0.0951625819640404268358, 1e-16, |x| x.cdf(1.0));
        test_case(1.0, 0.6321205588285576784045, |x| x.cdf(1.0));
        test_case(10.0, 0.9999546000702375151485, |x| x.cdf(1.0));
        test_case(f64::INFINITY, 1.0, |x| x.cdf(1.0));
        test_case(0.1, 1.0, |x| x.cdf(f64::INFINITY));
        test_case(1.0, 1.0, |x| x.cdf(f64::INFINITY));
        test_case(10.0, 1.0, |x| x.cdf(f64::INFINITY));
        test_case(f64::INFINITY, 1.0, |x| x.cdf(f64::INFINITY));
    }

    #[test]
    #[should_panic]
    fn test_neg_cdf() {
        get_value(0.1, |x| x.cdf(-1.0));
    }
}
