use std::f64;
use std::sync::{Once, ONCE_INIT};
use function::gamma;

/// The maximum factorial representable
/// by a 64-bit floating point without
/// overflowing
pub const MAX_ARG: u64 = 170;

/// Computes the factorial function `x -> x!` for
/// `170 >= x >= 0`. All factorials larger than `170!`
/// will overflow an `f64`. 
///
/// # Remarks
///
/// Returns `f64::INFINITY` if `x > 170`
pub fn factorial(x: u64) -> f64 {
    if x > MAX_ARG {
        f64::INFINITY
    } else {
        get_fcache()[x as usize]
    }
}

/// Computes the logarithmic factorial function `x -> ln(x!)`
/// for `x >= 0`. 
pub fn ln_factorial(x: u64) -> f64 {
    if x <= 1 {
        0.0
    } else if x > MAX_ARG {
        gamma::ln_gamma(x as f64 + 1.0)
    } else {
        get_fcache()[x as usize].ln()
    }
}

/// Computes the binomial coefficient `n choose k`
/// where `k` and `n` are non-negative values
pub fn binomial(n: u64, k: u64) -> f64 {
    if k > n {
        0.0
    } else {
        (0.5 + (ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)).exp()).floor()
    }
}

/// Computes the natural logarithm of the binomial coefficient
/// `ln(n choose k)` where `k` and `n` are non-negative values
pub fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        0.0
    } else {
        ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
    }
}

// Initialization for pre-computed cache of 171 factorial
// values 0!...170!
const CACHE_SIZE: usize = 171;

static mut FCACHE: &'static mut [f64; CACHE_SIZE] = &mut [1.0; CACHE_SIZE];
static START: Once = ONCE_INIT;

fn get_fcache() -> &'static [f64; CACHE_SIZE] {
    unsafe {
        START.call_once(|| {
            (1..CACHE_SIZE).fold(FCACHE[0], |acc, i| {
                let fac = acc * i as f64;
                FCACHE[i] = fac;
                fac
            });
        });
        FCACHE
    }
}
