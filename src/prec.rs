pub const DEFAULT_PREC: f64 = 0.0000000000000011102230246251565;

pub fn almost_eq(a: f64, b: f64, prec: f64) -> bool {
    // only true if a and b are infinite with same
    // sign
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }

    // NANs are never equal
    if a.is_nan() && b.is_nan() {
        return false;
    }

    (a - b).abs() < prec
}
