// polynomial evaluates a polynomial at z where coeff are the coeffecients
// to a polynomial of order k where k is the length of coeff and the coeffecient
// to the kth power is the kth element in coeff. E.g. [3,-1,2] equates to
// 2z^2 - z + 3
pub fn polynomial<'a>(z: f64, coeff: &'a [f64]) -> f64 {
    let mut sum = coeff[coeff.len() - 1];
    for x in coeff.into_iter().take(coeff.len() - 1) {
        sum = sum * z;
        sum = sum + x;
    }
    sum
}
