use std::f64;
use std::option::Option;
use rand::Rng;
use consts;
use distribution::{Distribution, Univariate, Continuous};
use error::StatsError;
use function::erf;
use result;

pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> result::Result<Normal> {
        if mean.is_nan() || std_dev.is_nan() || std_dev <= 0.0 {
            return Err(StatsError::BadParams);
        }
        Ok(Normal {
            mu: mean,
            sigma: std_dev,
        })
    }
}

impl Distribution for Normal {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.mu, self.sigma)
    }
}

impl Univariate for Normal {
    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    fn std_dev(&self) -> f64 {
        self.sigma
    }

    fn entropy(&self) -> f64 {
        self.sigma.ln() + consts::LN_SQRT_2PIE
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn median(&self) -> Option<f64> {
        Some(self.mu)
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        Ok(0.5 * erf::erfc((self.mu - x) / (self.sigma * f64::consts::SQRT_2)))
    }
}

impl Continuous for Normal {
    fn mode(&self) -> f64 {
        self.mu
    }

    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        let d = (x - self.mu) / self.sigma;
        (-0.5 * d * d).exp() / (consts::SQRT_2PI * self.sigma)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let d = (x - self.mu) / self.sigma;
        (-0.5 * d * d) - consts::LN_SQRT_2PI - self.sigma.ln()
    }
}

/// sample_unchecked draws a sample from a normal distribution using
/// the box-muller algorithm
pub fn sample_unchecked<R: Rng>(r: &mut R, mean: f64, std_dev: f64) -> f64 {
    let mut tuple = polar_transform(r.next_f64(), r.next_f64());
    while !tuple.2 {
        tuple = polar_transform(r.next_f64(), r.next_f64());
    }
    mean + std_dev * tuple.0
}

fn polar_transform(a: f64, b: f64) -> (f64, f64, bool) {
    let v1 = 2.0 * a - 1.0;
    let v2 = 2.0 * b - 1.0;
    let r = v1 * v2 + v2 * v2;
    if r >= 1.0 || r == 0.0 {
        return (0.0, 0.0, false);
    }

    let fac = (-2.0 * r.ln() / r).sqrt();
    (v1 * fac, v2 * fac, true)
}

#[cfg(test)]
mod test {
    use std::f64;
    use std::option::Option;
    use distribution::{Univariate, Continuous};
    use prec;
    use result;
    use super::Normal;
    
    fn try_create(mean: f64, std_dev: f64) -> Normal {
        let n = Normal::new(mean, std_dev);
        assert!(n.is_ok());
        n.unwrap()
    }
    
    fn create_case(mean: f64, std_dev: f64) {
        let n = try_create(mean, std_dev);
        assert_eq!(mean, n.mean());
        assert_eq!(std_dev, n.std_dev());
    }
    
    fn bad_create_case(mean: f64, std_dev: f64) {
        let n = Normal::new(mean, std_dev);
        assert!(n.is_err());
    }
    
    fn test_case<F>(mean: f64, std_dev: f64, expected: f64, eval: F) 
        where F : Fn(Normal) -> f64 {
    
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert_eq!(expected, x);        
    }
    
    fn test_almost<F>(mean: f64, std_dev: f64, expected: f64, acc: f64, eval: F) 
        where F : Fn(Normal) -> f64 {
        
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));    
    }
    
    fn test_optional<F>(mean: f64, std_dev: f64, expected: f64, eval: F)
        where F : Fn(Normal) -> Option<f64> {
    
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert!(x.is_some());
        
        let v = x.unwrap();
        assert_eq!(expected, v);   
    }
    
    fn test_result<F>(mean: f64, std_dev: f64, expected: f64, eval: F)
        where F : Fn(Normal) -> result::Result<f64> {
     
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert!(x.is_ok());
        
        let v = x.unwrap();
        assert_eq!(expected, v);       
    }
    
    fn test_result_almost<F>(mean: f64, std_dev: f64, expected: f64, acc: f64, eval: F)
        where F : Fn(Normal) -> result::Result<f64> {
            
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert!(x.is_ok());
        
        let v = x.unwrap();
        assert!(prec::almost_eq(expected, v, acc));  
    }
    
    #[test]
    fn test_create() {
        create_case(10.0, 0.1);
        create_case(-5.0, 1.0);
        create_case(0.0, 10.0);
        create_case(10.0, 100.0);
        create_case(-5.0, f64::INFINITY);
    }
    
    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 0.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, -1.0);
    }
    
    #[test]
    fn test_variance() {
        // note: mean is irrelevant to the variance
        // calculation, ergo all the test cases are
        // instantiated with the same mean
        test_case(0.0, 0.1, 0.1 * 0.1, |x| x.variance());
        test_case(0.0, 1.0, 1.0, |x| x.variance());
        test_case(0.0, 10.0, 100.0, |x| x.variance());
        test_case(0.0, f64::INFINITY, f64::INFINITY, |x| x.variance());
    }
    
    #[test]
    fn test_entropy() {
        // note: mean is irrelevant to the entropy calculation
        test_almost(0.0, 0.1, -0.8836465597893729422377, 1e-15, |x| x.entropy());
        test_case(0.0, 1.0, 1.41893853320467274178, |x| x.entropy());
        test_case(0.0, 10.0, 3.721523626198718425798, |x| x.entropy());
        test_case(0.0, f64::INFINITY, f64::INFINITY, |x| x.entropy());
    }
    
    #[test]
    fn test_skewness() {
        test_case(0.0, 0.1, 0.0, |x| x.skewness());
        test_case(4.0, 1.0, 0.0, |x| x.skewness());
        test_case(0.3, 10.0, 0.0, |x| x.skewness());
        test_case(0.0, f64::INFINITY, 0.0, |x| x.skewness());
    }
    
    #[test]
    fn test_mode() {
        // note: std_dev is irrelevant to the mode of a
        // normal distribution
        test_case(-0.0, 1.0, 0.0, |x| x.mode());
        test_case(0.0, 1.0, 0.0, |x| x.mode());
        test_case(0.1, 1.0, 0.1, |x| x.mode());
        test_case(1.0, 1.0, 1.0, |x| x.mode());
        test_case(-10.0, 1.0, -10.0, |x| x.mode());
        test_case(f64::INFINITY, 1.0, f64::INFINITY, |x| x.mode());
    }
    
    #[test]
    fn test_median() {
        // note: std_dev is irrelevant to the median of a
        // normal distribution
        test_optional(-0.0, 1.0, 0.0, |x| x.median());
        test_optional(0.0, 1.0, 0.0, |x| x.median());
        test_optional(0.1, 1.0, 0.1, |x| x.median());
        test_optional(1.0, 1.0, 1.0, |x| x.median());
        test_optional(-0.0, 1.0, -0.0, |x| x.median());
        test_optional(f64::INFINITY, 1.0, f64::INFINITY, |x| x.median());
    }
    
    #[test]
    fn test_min_max() {
        test_case(0.0, 0.1, f64::NEG_INFINITY, |x| x.min());
        test_case(-3.0, 10.0, f64::NEG_INFINITY, |x| x.min());
        test_case(0.0, 0.1, f64::INFINITY, |x| x.max());
        test_case(-3.0, 10.0, f64::INFINITY, |x| x.max());
    }
    
    #[test]
    fn test_pdf() {
        test_almost(10.0, 0.1, 5.530709549844416159162E-49, 1e-64, |x| x.pdf(8.5));
        test_almost(10.0, 0.1, 0.5399096651318805195056, 1e-14, |x| x.pdf(9.8));
        test_almost(10.0, 0.1, 3.989422804014326779399, 1e-15, |x| x.pdf(10.0));
        test_almost(10.0, 0.1, 0.5399096651318805195056, 1e-14, |x| x.pdf(10.2));
        test_almost(10.0, 0.1, 5.530709549844416159162E-49, 1e-64, |x| x.pdf(11.5));
        test_case(-5.0, 1.0, 1.486719514734297707908E-6, |x| x.pdf(-10.0));
        test_case(-5.0, 1.0, 0.01752830049356853736216, |x| x.pdf(-7.5));
        test_almost(-5.0, 1.0, 0.3989422804014326779399, 1e-16, |x| x.pdf(-5.0));
        test_case(-5.0, 1.0, 0.01752830049356853736216, |x| x.pdf(-2.5));
        test_case(-5.0, 1.0, 1.486719514734297707908E-6, |x| x.pdf(0.0));
        test_case(0.0, 10.0, 0.03520653267642994777747, |x| x.pdf(-5.0));
        test_almost(0.0, 10.0, 0.03866681168028492069412, 1e-17, |x| x.pdf(-2.5));
        test_almost(0.0, 10.0, 0.03989422804014326779399, 1e-17, |x| x.pdf(0.0));
        test_almost(0.0, 10.0, 0.03866681168028492069412, 1e-17, |x| x.pdf(2.5));
        test_case(0.0, 10.0, 0.03520653267642994777747, |x| x.pdf(5.0));
        test_almost(10.0, 100.0, 4.398359598042719404845E-4, 1e-19, |x| x.pdf(-200.0));
        test_case(10.0, 100.0, 0.002178521770325505313831, |x| x.pdf(-100.0));
        test_case(10.0, 100.0, 0.003969525474770117655105, |x| x.pdf(0.0));
        test_almost(10.0, 100.0, 0.002660852498987548218204, 1e-18, |x| x.pdf(100.0));
        test_case(10.0, 100.0, 6.561581477467659126534E-4, |x| x.pdf(200.0));
        test_case(-5.0, f64::INFINITY, 0.0, |x| x.pdf(-5.0));
        test_case(-5.0, f64::INFINITY, 0.0, |x| x.pdf(0.0));
        test_case(-5.0, f64::INFINITY, 0.0, |x| x.pdf(100.0));
    }
    
    #[test]
    fn test_ln_pdf() {
        test_almost(10.0, 0.1, (5.530709549844416159162E-49f64).ln(), 1e-13, |x| x.ln_pdf(8.5));
        test_almost(10.0, 0.1, (0.5399096651318805195056f64).ln(), 1e-13, |x| x.ln_pdf(9.8));
        test_almost(10.0, 0.1, (3.989422804014326779399f64).ln(), 1e-15, |x| x.ln_pdf(10.0));
        test_almost(10.0, 0.1, (0.5399096651318805195056f64).ln(), 1e-13, |x| x.ln_pdf(10.2));
        test_almost(10.0, 0.1, (5.530709549844416159162E-49f64).ln(), 1e-13, |x| x.ln_pdf(11.5));
        test_case(-5.0, 1.0, (1.486719514734297707908E-6f64).ln(), |x| x.ln_pdf(-10.0));
        test_case(-5.0, 1.0, (0.01752830049356853736216f64).ln(), |x| x.ln_pdf(-7.5));
        test_almost(-5.0, 1.0, (0.3989422804014326779399f64).ln(), 1e-15, |x| x.ln_pdf(-5.0));
        test_case(-5.0, 1.0, (0.01752830049356853736216f64).ln(), |x| x.ln_pdf(-2.5));
        test_case(-5.0, 1.0, (1.486719514734297707908E-6f64).ln(), |x| x.ln_pdf(0.0));
        test_case(0.0, 10.0, (0.03520653267642994777747f64).ln(), |x| x.ln_pdf(-5.0));
        test_case(0.0, 10.0, (0.03866681168028492069412f64).ln(), |x| x.ln_pdf(-2.5));
        test_case(0.0, 10.0, (0.03989422804014326779399f64).ln(), |x| x.ln_pdf(0.0));
        test_case(0.0, 10.0, (0.03866681168028492069412f64).ln(), |x| x.ln_pdf(2.5));
        test_case(0.0, 10.0, (0.03520653267642994777747f64).ln(), |x| x.ln_pdf(5.0));
        test_case(10.0, 100.0, (4.398359598042719404845E-4f64).ln(), |x| x.ln_pdf(-200.0));
        test_case(10.0, 100.0, (0.002178521770325505313831f64).ln(), |x| x.ln_pdf(-100.0));
        test_almost(10.0, 100.0, (0.003969525474770117655105f64).ln(), 1e-15, |x| x.ln_pdf(0.0));
        test_almost(10.0, 100.0, (0.002660852498987548218204f64).ln(), 1e-15, |x| x.ln_pdf(100.0));
        test_almost(10.0, 100.0, (6.561581477467659126534E-4f64).ln(), 1e-15, |x| x.ln_pdf(200.0));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(100.0));
    }
    
    #[test]
    fn test_cdf() {
        test_result(5.0, 2.0, 0.0, |x| x.cdf(f64::NEG_INFINITY));
        test_result_almost(5.0, 2.0, 0.0000002866515718, 1e-16, |x| x.cdf(-5.0));
        test_result_almost(5.0, 2.0, 0.0002326290790, 1e-13, |x| x.cdf(-2.0));
        test_result_almost(5.0, 2.0, 0.006209665325, 1e-12, |x| x.cdf(0.0));
        test_result(5.0, 2.0, 0.30853753872598689636229538939166226011639782444542207, |x| x.cdf(4.0));
        test_result(5.0, 2.0, 0.5, |x| x.cdf(5.0));
        test_result(5.0, 2.0, 0.69146246127401310363770461060833773988360217555457859, |x| x.cdf(6.0));
        test_result_almost(5.0, 2.0, 0.993790334674, 1e-12, |x| x.cdf(10.0));
    }
}