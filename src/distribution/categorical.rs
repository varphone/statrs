use std::f64;
use rand::Rng;
use {Result, StatsError};

/// Implements the [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution)
/// distribution, also known as the generalized Bernoulli or discrete distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Categorical, Discrete};
/// use statrs::statistics::Mode;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Categorical<R>
    where R: Rng
{
    norm_pmf: Vec<f64>,
    cdf: Vec<f64>,
    r: R,
}

impl<R> Categorical<R>
    where R: Rng
{
    pub fn new(prob_mass: &[f64], r: R) -> Result<Categorical<R>>
        where R: Rng
    {
        if !is_valid_prob_mass(prob_mass) {
            Err(StatsError::BadParams)
        } else {
            // extract un-normalized cdf
            let mut cdf = vec![0.0; prob_mass.len()];
            cdf[0] = prob_mass[0];
            for i in 1..prob_mass.len() {
                unsafe {
                    let val = cdf.get_unchecked(i - 1) + prob_mass.get_unchecked(i);
                    let elem = cdf.get_unchecked_mut(i);
                    *elem = val;
                }
            }

            // extract normalized probability mass
            let sum = cdf[cdf.len() - 1];
            let mut norm_pmf = vec![0.0; prob_mass.len()];
            for i in 0..prob_mass.len() {
                unsafe {
                    let elem = norm_pmf.get_unchecked_mut(i);
                    *elem = prob_mass.get_unchecked(i) / sum;
                }
            }
            Ok(Categorical {
                norm_pmf: norm_pmf,
                cdf: cdf,
                r: r,
            })
        }
    }
}

// determines if `p` is a valid probability mass array
// for the Categorical distribution
fn is_valid_prob_mass(p: &[f64]) -> bool {
    !p.iter().any(|&x| x < 0.0 || x.is_nan()) && !p.iter().all(|&x| x == 0.0)
}

#[test]
fn test_is_valid_prob_mass() {
    let invalid = [1.0, f64::NAN, 3.0];
    assert!(!is_valid_prob_mass(&invalid));
    let invalid2 = [-2.0, 5.0, 1.0, 6.2];
    assert!(!is_valid_prob_mass(&invalid2));
    let invalid3 = [0.0, 0.0, 0.0];
    assert!(!is_valid_prob_mass(&invalid3));
    let invalid4: [f64; 0] = [];
    assert!(!is_valid_prob_mass(&invalid4));
    let valid = [5.2, 0.00001, 1e-15, 1000000.12];
    assert!(is_valid_prob_mass(&valid));
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use rand::StdRng;
    use distribution::Categorical;

    fn try_create(prob_mass: &[f64]) -> Categorical<StdRng> {
        let r = StdRng::new().unwrap();
        let n = Categorical::new(prob_mass, r);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(prob_mass: &[f64]) {
        try_create(prob_mass);
    }

    fn bad_create_case(prob_mass: &[f64]) {
        let r = StdRng::new().unwrap();
        let n = Categorical::new(prob_mass, r);
        assert!(n.is_err());
    }

    #[test]
    fn test_create() {
        create_case(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(&[-1.0, 1.0]);
        bad_create_case(&[0.0, 0.0]);
    }
}