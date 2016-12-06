use result::Result;
use error::StatsError;
use consts;

/// Computes the generalized Exponential Integral function
/// where `x` is the argument and `n` is the integer power of the
/// denominator term.
///
/// # Errors
///
/// Returns an error if `x < 0.0` or the computation could not
/// converge after 100 iterations
///
/// # Remarks
///
/// This implementation follows the derivation in
/// <br />
/// <div>
/// <i>"Handbook of Mathematical Functions, Applied Mathematics Series, Volume 55"</i> - Abramowitz, M., and Stegun, I.A 1964
/// </div>
/// AND
/// <br />
/// <div>
/// <i>"Advanced mathematical methods for scientists and engineers" - Bender, Carl M.; Steven A. Orszag (1978). page 253
/// </div>
/// <br />
/// The continued fraction approac is used for `x > 1.0` while the taylor series expansions
/// is used for `0.0 < x <= 1`
///
/// # Examples
///
/// ```
/// ```
fn integral(x: f64, n: u64) -> Result<f64> {
    let eps = 0.00000000000000001;
    let max_iter = 100;
    let nf64 = n as f64;
    let near_f64min = 1e-100; // needs very small value that is not quite as small as f64 min

    // special cases
    if n == 0 {
        return Ok((-1.0 * x).exp() / x);
    }
    if x == 0.0 {
        return Ok(1.0 / (nf64 - 1.0));
    }

    if x > 1.0 {
        let mut b = x + nf64;
        let mut c = 1.0 / near_f64min;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..max_iter + 1 {
            let a = -1.0 * i as f64 * (nf64 - 1.0) + i as f64;
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < eps {
                return Ok(h * (-x).exp());
            }
        }
        Err(StatsError::ComputationFailedToConverge)
    } else {
        let mut factorial = 1.0;
        let mut result = if nf64 - 1.0 != 0.0 {
            1.0 / (nf64 - 1.0)
        } else {
            -1.0 * x.ln() - consts::EULER_MASCHERONI
        };
        for i in 1..max_iter + 1 {
            factorial *= -1.0 * x / i as f64;
            let del = if i as f64 != nf64 - 1.0 {
                -factorial / (i as f64 - nf64 + 1.0)
            } else {
                let mut psi = -1.0 * consts::EULER_MASCHERONI;
                for ii in 1..(nf64 as usize) {
                    psi += 1.0 / ii as f64;
                }
                factorial * (-1.0 * x.ln() + psi)
            };
            result += del;
            if del.abs() < result.abs() * eps {
                return Ok(result);
            }
        }
        Err(StatsError::ComputationFailedToConverge)
    }
}