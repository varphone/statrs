use function::internal;

/// Calculates a numerically stable `exp(x) - 1`
pub fn exp_minus_one(pow: f64) -> f64 {
    let x = pow.abs();
    if x > 0.1 {
        pow.exp() - 1.0
    } else {
        let mut k = 0;
        let mut term = 1.0;
        internal::series(|| {
            k += 1;
            term *= pow;
            term /= k as f64;
            term
        })
    }
}
