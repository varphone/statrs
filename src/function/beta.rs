//! Provides special functions related to the [beta](https://en.wikipedia.org/wiki/Beta_function)
//! function

use std::f64;
use error::StatsError;
use function::gamma;
use prec;

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn ln_beta(a: f64, b: f64) -> f64 {
    assert!(a > 0.0, format!("{}", StatsError::ArgMustBePositive("a")));
    assert!(b > 0.0, format!("{}", StatsError::ArgMustBePositive("b")));
    gamma::ln_gamma(a) + gamma::ln_gamma(b) - gamma::ln_gamma(a + b)
}

/// Computes the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter.
///
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Panics
///
/// If `a < 0.0`, `b < 0.0`, `x < 0.0`, or `x > 1.0`
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    beta_reg(a, b, x) * beta(a, b)
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Panics
///
/// if `a < 0.0`, `b < 0.0`, `x < 0.0`, or `x > 1.0`
pub fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    assert!(a >= 0.0, format!("{}", StatsError::ArgNotNegative("a")));
    assert!(b >= 0.0, format!("{}", StatsError::ArgNotNegative("b")));
    assert!(x >= 0.0 && x <= 1.0,
            format!("{}", StatsError::ArgIntervalIncl("x", 0.0, 1.0)));

    let bt = match x {
        0.0 | 1.0 => 0.0,
        _ => {
            (gamma::ln_gamma(a + b) - gamma::ln_gamma(a) - gamma::ln_gamma(b) + a * x.ln() +
             b * (1.0 - x).ln())
                .exp()
        }
    };
    let symm_transform = x >= (a + 1.0) / (a + b + 2.0);
    let eps = prec::F64_PREC;
    let fpmin = f64::MIN_POSITIVE / eps;

    let mut a = a;
    let mut b = b;
    let mut x = x;
    if symm_transform {
        let swap = a;
        x = 1.0 - x;
        a = b;
        b = swap;
    }

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..141 {
        let m = m as f64;
        let m2 = m * 2.0;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;

        if d.abs() < fpmin {
            d = fpmin;
        }

        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }

        d = 1.0 / d;
        h = h * d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;

        if d.abs() < fpmin {
            d = fpmin;
        }

        c = 1.0 + aa / c;

        if c.abs() < fpmin {
            c = fpmin;
        }

        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() <= eps {
            return if symm_transform {
                1.0 - bt * h / a
            } else {
                bt * h / a
            };
        }
    }

    if symm_transform {
        1.0 - bt * h / a
    } else {
        bt * h / a
    }
}
