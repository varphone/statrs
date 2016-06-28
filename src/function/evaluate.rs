/// evaluates a polynomial at `z` where `coeff` are the coeffecients
/// to a polynomial of order `k` where `k` is the length of `coeff` and the coeffecient
/// to the `k`th power is the `k`th element in coeff. E.g. [3,-1,2] equates to
/// `2z^2 - z + 3`
pub fn polynomial(z: f64, coeff: &[f64]) -> f64 {
    let mut sum = coeff[coeff.len() - 1];
    for i in (0..coeff.len() - 1).rev() {
        sum *= z;
        sum += coeff[i];
    }
    sum
}

/// Evaluates a numerically stable
/// series summation where `next` returns the
/// next summand in the series
pub fn series<F>(mut next: F) -> f64
    where F: FnMut() -> f64
{
    let factor = (1 << 16) as f64;
    let mut comp = 0.0;
    let mut sum = next();

    loop {
        let cur = next();
        let y = cur - comp;
        let t = sum + y;
        comp = t - sum - y;
        sum = t;
        if sum.abs() >= (factor * cur).abs() {
            break;
        }
    }
    sum
}
