use std::f64;
use std::option::Option;
use rand::Rng;
use consts;
use distribution::{Distribution, Univariate, Continuous};
use error::StatsError;
use functions::erf;
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

pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    pub fn new(mean: f64, std_dev: f64) -> result::Result<LogNormal> {
        if mean.is_nan() || std_dev.is_nan() || std_dev <= 0.0 {
            return Err(StatsError::BadParams);
        }
        Ok(LogNormal {
            mu: mean,
            sigma: std_dev,
        })
    }
}

impl Distribution for LogNormal {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.mu, self.sigma).exp()
    }
}

impl Univariate for LogNormal {
    fn mean(&self) -> f64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp()
    }

    fn variance(&self) -> f64 {
        let sigma2 = self.sigma * self.sigma;
        (sigma2.exp() - 1.0) * (self.mu + self.mu + sigma2).exp()
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        0.5 + self.sigma.ln() + self.mu + consts::LN_SQRT_2PI
    }

    fn skewness(&self) -> f64 {
        let expsigma2 = (self.sigma * self.sigma).exp();
        (expsigma2 + 2.0) * (expsigma2 - 1.0).sqrt()
    }

    fn median(&self) -> Option<f64> {
        Some(self.mu.exp())
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        if x < 0.0 {
            Ok(0.0)
        } else {
            Ok(0.5 * erf::erfc((self.mu - x.ln()) / (self.sigma * f64::consts::SQRT_2)))
        }
    }
}

impl Continuous for LogNormal {
    fn mode(&self) -> f64 {
        (self.mu - self.sigma * self.sigma).exp()
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        match x {
            0.0 => 0.0,
            _ => {
                let d = (x.ln() - self.mu) / self.sigma;
                (-0.5 * d * d).exp() / (x * consts::SQRT_2PI * self.sigma)
            }
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        match x {
            0.0 => f64::NEG_INFINITY,
            _ => {
                let d = (x.ln() - self.mu) / self.sigma;
                (-0.5 * d * d) - consts::LN_SQRT_2PI - (x * self.sigma).ln()
            }
        }
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
    use super::{Normal, LogNormal};
    
    fn try_create(mean: f64, std_dev: f64) -> Normal {
        let n = Normal::new(mean, std_dev);
        assert!(n.is_ok());
        n.unwrap()
    }
    
    fn try_create_log(mean: f64, std_dev: f64) -> LogNormal {
        let n = LogNormal::new(mean, std_dev);
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
    
    fn bad_create_log_case(mean: f64, std_dev: f64) {
        let n = LogNormal::new(mean, std_dev);
        assert!(n.is_err());
    }
    
    fn test_case<F>(mean: f64, std_dev: f64, expected: f64, eval: F) 
        where F : Fn(Normal) -> f64 {
    
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert_eq!(expected, x);        
    }
    
    fn test_log_case<F>(mean: f64, std_dev: f64, expected: f64, eval: F)
        where F : Fn(LogNormal) -> f64 {
    
        let n = try_create_log(mean, std_dev);
        let x = eval(n);
        assert_eq!(expected, x);         
    }
    
    fn test_almost<F>(mean: f64, std_dev: f64, expected: f64, acc: f64, eval: F) 
        where F : Fn(Normal) -> f64 {
        
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc));    
    }
    
    fn test_log_almost<F>(mean: f64, std_dev: f64, expected: f64, acc: f64, eval: F)
        where F : Fn(LogNormal) -> f64 {
        
        let n = try_create_log(mean, std_dev);
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
    
    #[test]
    fn test_create_log() {
        try_create_log(10.0, 0.1);
        try_create_log(-5.0, 1.0);
        try_create_log(0.0, 10.0);
        try_create_log(10.0, 100.0);
        try_create_log(-5.0, f64::INFINITY);
    }
    
    #[test]
    fn test_bad_create_log() {
        bad_create_log_case(0.0, 0.0);
        bad_create_log_case(f64::NAN, 1.0);
        bad_create_log_case(1.0, f64::NAN);
        bad_create_log_case(f64::NAN, f64::NAN);
        bad_create_log_case(1.0, -1.0);
    }
    
    #[test]
    fn test_log_mean() {
        test_log_case(-1.0, 0.1, 0.369723444544058982601, |x| x.mean());
        test_log_case(-1.0, 1.5, 1.133148453066826316829, |x| x.mean());
        test_log_case(-1.0, 2.5, 8.372897488127264663205, |x| x.mean());
        test_log_case(-1.0, 5.5, 1362729.18425285481771, |x| x.mean());
        test_log_case(-0.1, 0.1, 0.9093729344682314204933, |x| x.mean());
        test_log_case(-0.1, 1.5, 2.787095460565850768514, |x| x.mean());
        test_log_case(-0.1, 2.5, 20.59400471119602917533, |x| x.mean());
        test_log_almost(-0.1, 5.5, 3351772.941252693807591, 1e-9, |x| x.mean());
        test_log_case(0.1, 0.1, 1.110710610355705232259, |x| x.mean());
        test_log_case(0.1, 1.5, 3.40416608279081898632, |x| x.mean());
        test_log_almost(0.1, 2.5, 25.15357415581836182776, 1e-14, |x| x.mean());
        test_log_almost(0.1, 5.5, 4093864.715172665106863, 1e-8, |x| x.mean());
        test_log_almost(1.5, 0.1, 4.50415363028848413209, 1e-15, |x| x.mean());
        test_log_case(1.5, 1.5, 13.80457418606709491926, |x| x.mean());
        test_log_case(1.5, 2.5, 102.0027730826996844534, |x| x.mean());
        test_log_case(1.5, 5.5, 16601440.05723477471392, |x| x.mean());
        test_log_almost(2.5, 0.1, 12.24355896580102707724, 1e-14, |x| x.mean());
        test_log_case(2.5, 1.5, 37.52472315960099891407, |x| x.mean());
        test_log_case(2.5, 2.5, 277.2722845231339804081, |x| x.mean());
        test_log_case(2.5, 5.5, 45127392.83383337999291, |x| x.mean());
        test_log_almost(5.5, 0.1, 245.9184556788219446833, 1e-13, |x| x.mean());
        test_log_case(5.5, 1.5, 753.7042125545612656606, |x| x.mean());
        test_log_case(5.5, 2.5, 5569.162708566004074422, |x| x.mean());
        test_log_case(5.5, 5.5, 906407915.0111549133446, |x| x.mean());
    }
    
    #[test]
    fn test_log_variance() {
        // note: variance seems to be only accurate to around 15 orders
        // of magnitude. Hopefully in the future we can extend the precision
        // of this function
        test_log_almost(-1.0, 0.1, 0.001373811865368952608715, 1e-16, |x| x.variance());
        test_log_case(-1.0, 1.5, 10.898468544015731954, |x| x.variance());
        test_log_case(-1.0, 2.5, 36245.39726189994988081, |x| x.variance());
        test_log_almost(-1.0, 5.5, 2.5481629178024539E+25, 1e10, |x| x.variance());
        test_log_almost(-0.1, 0.1, 0.008311077467909703803238, 1e-16, |x| x.variance());
        test_log_case(-0.1, 1.5, 65.93189259328902509552, |x| x.variance());
        test_log_almost(-0.1, 2.5, 219271.8756420929704707, 1e-10, |x| x.variance());
        test_log_almost(-0.1, 5.5, 1.541548733459471E+26, 1e12, |x| x.variance());
        test_log_almost(0.1, 0.1, 0.01239867063063756838894, 1e-15, |x| x.variance());
        test_log_almost(0.1, 1.5, 98.35882573290010981464, 1e-13, |x| x.variance());
        test_log_almost(0.1, 2.5, 327115.1995809995715014, 1e-10, |x| x.variance());
        test_log_almost(0.1, 5.5, 2.299720473192458E+26, 1e12, |x| x.variance());
        test_log_almost(1.5, 0.1, 0.2038917589520099120699, 1e-14, |x| x.variance());
        test_log_almost(1.5, 1.5, 1617.476145997433210727, 1e-12, |x| x.variance());
        test_log_almost(1.5, 2.5, 5379293.910566451644527, 1e-9, |x| x.variance());
        test_log_almost(1.5, 5.5, 3.7818090853910142E+27, 1e12, |x| x.variance());
        test_log_almost(2.5, 0.1, 1.506567645006046841936, 1e-13, |x| x.variance());
        test_log_almost(2.5, 1.5, 11951.62198145717670088, 1e-11, |x| x.variance());
        test_log_case(2.5, 2.5, 39747904.47781154725843, |x| x.variance());
        test_log_almost(2.5, 5.5, 2.7943999487399818E+28, 1e13, |x| x.variance());
        test_log_almost(5.5, 0.1, 607.7927673399807484235, 1e-11, |x| x.variance());
        test_log_case(5.5, 1.5, 4821628.436260521100027, |x| x.variance());
        test_log_case(5.5, 2.5, 16035449147.34799637823, |x| x.variance());
        test_log_case(5.5, 5.5, 1.127341399856331737823E+31, |x| x.variance());
    }
}