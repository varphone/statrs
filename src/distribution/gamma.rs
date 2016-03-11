use std::f64;
use std::option::Option;
use rand::Rng;
use consts;
use distribution::{Distribution, Univariate, Continuous};
use distribution::normal;
use functions::gamma;
use result;

pub struct Gamma {
    a: f64,
    b: f64,
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> result::Result<Gamma> {
        if shape < 0.0 || rate < 0.0 {
            return Err(consts::BAD_DISTR_PARAMS.to_string());
        }
        Ok(Gamma {
            a: shape,
            b: rate,
        })
    }
}

impl Distribution for Gamma {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.a, self.b)
    }
}

impl Univariate for Gamma {
    fn mean(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => self.a,
            (_, _) => self.a / self.b,
        }
    }

    fn variance(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => 0.0,
            (_, _) => self.a / (self.b * self.b),
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => 0.0,
            (_, _) => {
                self.a - self.b.ln() + gamma::gamma_ln(self.a) +
                (1.0 - self.a) * gamma::digamma(self.a)
            }
        }
    }

    fn skewness(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => 0.0,
            (_, _) => 2.0 / self.a.sqrt(),
        }
    }

    fn median(&self) -> Option<f64> {
        None
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        match (self.a, self.b) {
            (0.0, 0.0) => Ok(0.0),
            (_, f64::INFINITY) => {
                if x == self.a {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            }
            (_, _) => gamma::gamma_lr(self.a, x * self.b),
        }
    }
}

impl Continuous for Gamma {
    fn mode(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => self.a,
            (_, _) => (self.a - 1.0) / self.b,
        }
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => 0.0,
            (_, f64::INFINITY) => {
                if x == self.a {
                    f64::INFINITY
                } else {
                    0.0
                }
            }
            (1.0, _) => self.b * (-self.b * x).exp(),
            (_, _) if self.a > 160.0 => self.ln_pdf(x).exp(),
            (_, _) => {
                self.b.powf(self.a) * x.powf(self.a - 1.0) * (-self.b * x).exp() /
                gamma::gamma(self.a)
            }
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NEG_INFINITY,
            (_, f64::INFINITY) => {
                if x == self.a {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            }
            (1.0, _) => self.b.ln() - self.b * x,
            (_, _) => {
                self.a * self.b.ln() + (self.a - 1.0) * x.ln() - self.b * x -
                gamma::gamma_ln(self.a)
            }
        }
    }
}

fn sample_unchecked<R: Rng>(r: &mut R, shape: f64, rate: f64) -> f64 {
    if rate == f64::INFINITY {
        return shape;
    }

    let mut a = shape;
    let mut afix = 1.0;

    if shape < 1.0 {
        a = shape + 1.0;
        afix = r.next_f64().powf(1.0 / shape);
    }

    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x = normal::sample_unchecked(r, 0.0, 1.0);
        let mut v = 1.0 + c * x;
        while v <= 0.0 {
            x = normal::sample_unchecked(r, 0.0, 1.0);
            v = 1.0 + c * x;
        }

        v = v * v * v;
        x = x * x;
        let u = r.next_f64();
        if u < 1.0 - 0.0331 * x * x {
            return afix * d * v / rate;
        }
        if u.ln() < 0.5 * x + d * (1.0 - v - v.ln()) {
            return afix * d * v / rate;
        }
    }
}
