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

    fn entropy_case(min: f64, max: f64) {
        let n = try_create(min, max);
        assert_eq!(n.entropy(), (max - min).ln());
    }

    fn skewness_case(min: f64, max: f64) {
        let n = try_create(min, max);
        assert_eq!(n.skewness(), 0.0);
    }

    fn mode_case(min: f64, max: f64) {
        let n = try_create(min, max);
        assert_eq!(n.mode(), (min + max) / 2.0);
    }

    fn median_case(min: f64, max: f64) {
        let n = try_create(min, max);
        let r = n.median();
        assert!(r.is_some());

        let m = r.unwrap();
        assert_eq!(m, (min + max) / 2.0);
    }

    fn pdf_case(min: f64, max: f64) {
        let n = try_create(min, max);
        for i in 0..11 {
            let x = i as f64 - 5.0;
            if x >= min && x <= max {
                assert_eq!(n.pdf(x), 1.0 / (max - min));
            } else {
                assert_eq!(n.pdf(x), 0.0);
            }
        }
    }

    fn ln_pdf_case(min: f64, max: f64) {
        let n = try_create(min, max);
        for i in 0..11 {
            let x = i as f64 - 5.0;
            if x >= min && x <= max {
                assert_eq!(n.ln_pdf(x), -(max - min).ln());
            } else {
                assert_eq!(n.ln_pdf(x), f64::NEG_INFINITY);
            }
        }
    }

    fn cdf_case(min: f64, max: f64) {
        let n = try_create(min, max);
        for i in 0..11 {
            let x = i as f64 - 5.0;
            let r = n.cdf(x);
            assert!(r.is_ok());

            let v = r.unwrap();
            if x <= min {
                assert_eq!(v, 0.0);
            } else if x >= max {
                assert_eq!(v, 1.0);
            } else {
                assert_eq!(v, (x - min) / (max - min));
            }
        }
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
        entropy_case(-0.0, 2.0);
        entropy_case(0.0, 2.0);
        entropy_case(0.1, 4.0);
        entropy_case(1.0, 10.0);
        entropy_case(10.0, 11.0);
        entropy_case(0.0, f64::INFINITY);
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
        mode_case(-0.0, 2.0);
        mode_case(0.0, 2.0);
        mode_case(0.1, 4.0);
        mode_case(1.0, 10.0);
        mode_case(10.0, 11.0);
        mode_case(0.0, f64::INFINITY);
    }

    #[test]
    fn test_median() {
        median_case(-0.0, 2.0);
        median_case(0.0, 2.0);
        median_case(0.1, 4.0);
        median_case(1.0, 10.0);
        median_case(10.0, 11.0);
        median_case(0.0, f64::INFINITY);
    }

    #[test]
    fn test_pdf() {
        pdf_case(0.0, 0.0);
        pdf_case(0.0, 0.1);
        pdf_case(0.0, 1.0);
        pdf_case(0.0, 10.0);
        pdf_case(-5.0, 100.0);
        pdf_case(0.0, f64::INFINITY);
    }

    #[test]
    fn test_ln_pdf() {
        ln_pdf_case(0.0, 0.0);
        ln_pdf_case(0.0, 0.1);
        ln_pdf_case(0.0, 1.0);
        ln_pdf_case(0.0, 10.0);
        ln_pdf_case(-5.0, 100.0);
        ln_pdf_case(0.0, f64::INFINITY);
    }

    #[test]
    fn test_cdf() {
        cdf_case(0.0, 0.0);
        cdf_case(0.0, 0.1);
        cdf_case(0.0, 1.0);
        cdf_case(0.0, 10.0);
        cdf_case(-5.0, 100.0);
        cdf_case(0.0, f64::INFINITY);
    }
}
