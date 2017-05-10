/// Returns true if there are no elements in `x` in `arr`
/// such that `x <= 0.0` or `x` is `f64::NAN` and `sum(arr) > 0.0`.
/// IF `incl_zero` is true, it tests for `x < 0.0` instead of `x <= 0.0` 
pub fn is_valid_multinomial(arr: &[f64], incl_zero: bool) -> bool {
    let mut sum = 0.0;
    for i in 0..arr.len() {
        let el = *unsafe { arr.get_unchecked(i) };
        if incl_zero && el < 0.0 {
            return false;
        } else if !incl_zero && el <= 0.0 {
            return false;
        } else if el.is_nan() {
            return false;
        }
        sum += el;
    }
    if sum == 0.0 {
        false
    } else {
        true
    }
}

#[test]
fn test_is_valid_multinomial() {
    use std::f64;

    let invalid = [1.0, f64::NAN, 3.0];
    assert!(!is_valid_multinomial(&invalid, true));
    let invalid2 = [-2.0, 5.0, 1.0, 6.2];
    assert!(!is_valid_multinomial(&invalid2, true));
    let invalid3 = [0.0, 0.0, 0.0];
    assert!(!is_valid_multinomial(&invalid3, true));
    let valid = [5.2, 0.0, 1e-15, 1000000.12];
    assert!(is_valid_multinomial(&valid, true));
}

#[test]
fn test_is_valid_multinomial_no_zero() {
    let invalid = [5.2, 0.0, 1e-15, 1000000.12];
    assert!(!is_valid_multinomial(&invalid, false));
}