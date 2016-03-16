use std::f64::consts;
use std::option::Option;
use rand::Rng;
use distribution::{Distribution, Univariate, Continuous};
use error::StatsError;
use result;

pub struct Triangular {
    min: f64,
    max: f64,
    mode: f64,
}

impl Triangular {
    pub fn new(min: f64, max: f64, mode: f64) -> result::Result<Triangular> {
        if min.is_infinite() || max.is_infinite() || mode.is_infinite() {
            return Err(StatsError::BadParams);
        }
        if min.is_nan() || max.is_nan() || mode.is_nan() {
            return Err(StatsError::BadParams);
        }
        if max < mode || mode < min {
            return Err(StatsError::BadParams);
        }
        Ok(Triangular {
            min: min,
            max: max,
            mode: mode,
        })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn mode(&self) -> f64 {
        self.mode
    }
}

impl Distribution for Triangular {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.min, self.max, self.mode)
    }
}

impl Univariate for Triangular {
    fn mean(&self) -> f64 {
        (self.min + self.max + self.mode) / 3.0
    }

    fn variance(&self) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        0.5 + ((self.max - self.min) / 2.0).ln()
    }

    fn skewness(&self) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        let q = consts::SQRT_2 * (a + b - 2.0 * c) * (2.0 * a - b - c) * (a - 2.0 * b + c);
        let d = 5.0 * (a * a + b * b + c * c - a * b - a * c - b * c).powf(3.0 / 2.0);
        q / d
    }

    fn median(&self) -> Option<f64> {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if c >= (a + b) / 2.0 {
            Some(a + ((b - a) * (c - a) / 2.0).sqrt())
        } else {
            Some(b - ((b - a) * (b - c) / 2.0).sqrt())
        }
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if x < a {
            return Ok(0.0);
        }
        if a <= x && x <= c {
            return Ok((x - a) * (x - a) / ((b - a) * (c - a)));
        }
        if c < x && x <= b {
            return Ok(1.0 - (b - x) * (b - x) / ((b - a) * (b - c)));
        }
        Ok(1.0)
    }
}

impl Continuous for Triangular {
    fn mode(&self) -> f64 {
        self.mode
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }

    fn pdf(&self, x: f64) -> f64 {
        let a = self.min;
        let b = self.max;
        let c = self.mode;
        if a <= x && x <= c {
            return 2.0 * (x - a) / ((b - a) * (c - a));
        }
        if c < x && x <= b {
            return 2.0 * (b - x) / ((b - a) * (b - c));
        }
        0.0
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

fn sample_unchecked<R: Rng>(r: &mut R, min: f64, max: f64, mode: f64) -> f64 {
    let f = r.next_f64();
    if f < (mode - min) / (max - min) {
        min + (f * (max - min) * (mode - min)).sqrt()
    } else {
        max - ((1.0 - f) * (max - min) * (max - mode)).sqrt()
    }
}

#[cfg(test)]
mod test {
    use std::f64;
    use super::Triangular;

    fn try_create(min: f64, max: f64, mode: f64) -> Triangular {
        let r = Triangular::new(min, max, mode);
        assert!(r.is_ok());
        r.unwrap()
    }

    fn create_case(min: f64, max: f64, mode: f64) {
        let n = try_create(min, max, mode);
        assert_eq!(n.min(), min);
        assert_eq!(n.max(), max);
        assert_eq!(n.mode(), mode);
    }

    fn bad_create_case(min: f64, max: f64, mode: f64) {
        let n = Triangular::new(min, max, mode);
        assert!(n.is_err());
    }

    #[test]
    fn test_create() {
        create_case(0.0, 0.0, 0.0);
        create_case(-1.0, 1.0, 0.0);
        create_case(1.0, 2.0, 1.0);
        create_case(5.0, 25.0, 25.0);
        create_case(1.0e-5, 1.0e5, 1.0e-3);
        create_case(0.0, 1.0, 0.9);
        create_case(-4.0, -0.5, -2.0);
        create_case(-13.039, 8.42, 1.17);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 1.0, -0.1);
        bad_create_case(0.0, 1.0, 1.1);
        bad_create_case(0.0, -1.0, 0.5);
        bad_create_case(2.0, 1.0, 1.5);
        bad_create_case(f64::NAN, 1.0, 0.5);
        bad_create_case(0.2, f64::NAN, 0.5);
        bad_create_case(0.5, 1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN, f64::NAN);
        bad_create_case(f64::NEG_INFINITY, 1.0, 0.5);
        bad_create_case(0.0, f64::INFINITY, 0.5);
    }
}
