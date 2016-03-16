use std::f64;
use std::option::Option;
use rand::Rng;
use distribution::{Distribution, Univariate, Continuous};
use error::StatsError;
use result;

pub struct Uniform {
    min: f64,
    max: f64,
}

impl Uniform {
    pub fn new(min: f64, max: f64) -> result::Result<Uniform> {
        if min > max || min.is_nan() || max.is_nan() {
            return Err(StatsError::BadParams);
        }
        Ok(Uniform {
            min: min,
            max: max,
        })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Distribution for Uniform {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        self.min + r.next_f64() * (self.max - self.min)
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

    fn median(&self) -> Option<f64> {
        Some((self.min + self.max) / 2.0)
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        return if x <= self.min {
            Ok(0.0)
        } else if x >= self.max {
            Ok(1.0)
        } else {
            Ok((x - self.min) / (self.max - self.min))
        };
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
        return if x < self.min || x > self.max {
            0.0
        } else {
            1.0 / (self.max - self.min)
        };
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        return if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            -(self.max - self.min).ln()
        };
    }
}

#[cfg(test)]
mod test {
    use std::f64;
    use distribution::{Univariate, Continuous};
    use prec;
    use super::Uniform;

    fn try_create(min: f64, max: f64) -> Uniform {
        let r = Uniform::new(min, max);
        assert!(r.is_ok());
        r.unwrap()
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

    fn entropy_case(min: f64, max: f64, expected: f64) {
        let n = try_create(min, max);
        assert!(prec::almost_eq(expected, n.entropy(), 1e-15));
    }

    fn skewness_case(min: f64, max: f64) {
        let n = try_create(min, max);
        assert_eq!(0.0, n.skewness());
    }

    fn mode_case(min: f64, max: f64, expected: f64) {
        let n = try_create(min, max);
        assert_eq!(expected, n.mode());
    }

    fn median_case(min: f64, max: f64, expected: f64) {
        let n = try_create(min, max);
        let r = n.median();
        assert!(r.is_some());

        let m = r.unwrap();
        assert_eq!(expected, m);
    }

    fn pdf_case(min: f64, max: f64, input: f64, expected: f64) {
        let n = try_create(min, max);
        assert_eq!(expected, n.pdf(input));
    }

    fn ln_pdf_case(min: f64, max: f64, input: f64, expected: f64) {
        let n = try_create(min, max);
        assert!(prec::almost_eq(expected, n.ln_pdf(input), 1e-15));
    }

    fn cdf_case(min: f64, max: f64, input: f64, expected: f64) {
        let n = try_create(min, max);
        let r = n.cdf(input);
        assert!(r.is_ok());

        let m = r.unwrap();
        assert_eq!(expected, m);
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
    fn test_entropy() {
        entropy_case(-0.0, 2.0, 0.6931471805599453094172);
        entropy_case(0.0, 2.0, 0.6931471805599453094172);
        entropy_case(0.1, 4.0, 1.360976553135600743431);
        entropy_case(1.0, 10.0, 2.19722457733621938279);
        entropy_case(10.0, 11.0, 0.0);
        entropy_case(0.0, f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_skewness() {
        skewness_case(-0.0, 2.0);
        skewness_case(0.0, 2.0);
        skewness_case(0.1, 4.0);
        skewness_case(1.0, 10.0);
        skewness_case(10.0, 11.0);
        skewness_case(0.0, f64::INFINITY);
    }

    #[test]
    fn test_mode() {
        mode_case(-0.0, 2.0, 1.0);
        mode_case(0.0, 2.0, 1.0);
        mode_case(0.1, 4.0, 2.05);
        mode_case(1.0, 10.0, 5.5);
        mode_case(10.0, 11.0, 10.5);
        mode_case(0.0, f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_median() {
        median_case(-0.0, 2.0, 1.0);
        median_case(0.0, 2.0, 1.0);
        median_case(0.1, 4.0, 2.05);
        median_case(1.0, 10.0, 5.5);
        median_case(10.0, 11.0, 10.5);
        median_case(0.0, f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_pdf() {
        pdf_case(0.0, 0.0, -5.0, 0.0);
        pdf_case(0.0, 0.0, 0.0, f64::INFINITY);
        pdf_case(0.0, 0.0, 5.0, 0.0);
        pdf_case(0.0, 0.1, -5.0, 0.0);
        pdf_case(0.0, 0.1, 0.05, 10.0);
        pdf_case(0.0, 0.1, 5.0, 0.0);
        pdf_case(0.0, 1.0, -5.0, 0.0);
        pdf_case(0.0, 1.0, 0.5, 1.0);
        pdf_case(0.0, 0.1, 5.0, 0.0);
        pdf_case(0.0, 10.0, -5.0, 0.0);
        pdf_case(0.0, 10.0, 1.0, 0.1);
        pdf_case(0.0, 10.0, 5.0, 0.1);
        pdf_case(0.0, 10.0, 11.0, 0.0);
        pdf_case(-5.0, 100.0, -10.0, 0.0);
        pdf_case(-5.0, 100.0, -5.0, 0.009523809523809523809524);
        pdf_case(-5.0, 100.0, 0.0, 0.009523809523809523809524);
        pdf_case(-5.0, 100.0, 101.0, 0.0);
        pdf_case(0.0, f64::INFINITY, -5.0, 0.0);
        pdf_case(0.0, f64::INFINITY, 10.0, 0.0);
        pdf_case(0.0, f64::INFINITY, f64::INFINITY, 0.0);
    }

    #[test]
    fn test_ln_pdf() {
        ln_pdf_case(0.0, 0.0, -5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, 0.0, 0.0, f64::INFINITY);
        ln_pdf_case(0.0, 0.0, 5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, 0.1, -5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, 0.1, 0.05, 2.302585092994045684018);
        ln_pdf_case(0.0, 0.1, 5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, 1.0, -5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, 1.0, 0.5, 0.0);
        ln_pdf_case(0.0, 0.1, 5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, 10.0, -5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, 10.0, 1.0, -2.302585092994045684018);
        ln_pdf_case(0.0, 10.0, 5.0, -2.302585092994045684018);
        ln_pdf_case(0.0, 10.0, 11.0, f64::NEG_INFINITY);
        ln_pdf_case(-5.0, 100.0, -10.0, f64::NEG_INFINITY);
        ln_pdf_case(-5.0, 100.0, -5.0, -4.653960350157523371101);
        ln_pdf_case(-5.0, 100.0, 0.0, -4.653960350157523371101);
        ln_pdf_case(-5.0, 100.0, 101.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, f64::INFINITY, -5.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, f64::INFINITY, 10.0, f64::NEG_INFINITY);
        ln_pdf_case(0.0, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    }

    #[test]
    fn test_cdf() {
        cdf_case(0.0, 0.0, -5.0, 0.0);
        cdf_case(0.0, 0.0, 0.0, 0.0);
        cdf_case(0.0, 0.0, 5.0, 1.0);
        cdf_case(0.0, 0.1, -5.0, 0.0);
        cdf_case(0.0, 0.1, 0.05, 0.5);
        cdf_case(0.0, 0.1, 5.0, 1.0);
        cdf_case(0.0, 1.0, -5.0, 0.0);
        cdf_case(0.0, 1.0, 0.5, 0.5);
        cdf_case(0.0, 0.1, 5.0, 1.0);
        cdf_case(0.0, 10.0, -5.0, 0.0);
        cdf_case(0.0, 10.0, 1.0, 0.1);
        cdf_case(0.0, 10.0, 5.0, 0.5);
        cdf_case(0.0, 10.0, 11.0, 1.0);
        cdf_case(-5.0, 100.0, -10.0, 0.0);
        cdf_case(-5.0, 100.0, -5.0, 0.0);
        cdf_case(-5.0, 100.0, 0.0, 0.04761904761904761904762);
        cdf_case(-5.0, 100.0, 101.0, 1.0);
        cdf_case(0.0, f64::INFINITY, -5.0, 0.0);
        cdf_case(0.0, f64::INFINITY, 10.0, 0.0);
        cdf_case(0.0, f64::INFINITY, f64::INFINITY, 1.0);
    }
}
