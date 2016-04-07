use std::f64;
use std::option::Option;
use rand::Rng;
use distribution::{Distribution, Univariate, Continuous};
use distribution::normal;
use error::StatsError;
use function::gamma;
use result;

#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    a: f64,
    b: f64,
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> result::Result<Gamma> {
        let is_nan = shape.is_nan() || rate.is_nan();
        match (shape, rate, is_nan) {
            (_, _, true) => Err(StatsError::BadParams),
            (_, _, false) if shape < 0.0 || rate < 0.0 => Err(StatsError::BadParams),
            (_, _, false) => {
                Ok(Gamma {
                    a: shape,
                    b: rate,
                })
            }
        }
    }
    
    pub fn shape(&self) -> f64 {
        self.a
    }
    
    pub fn rate(&self) -> f64 {
        self.b
    }
}

impl Distribution for Gamma {
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        sample_unchecked(r, self.a, self.b)
    }
}

impl Univariate for Gamma {
    fn mean(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => self.a,
            (_, _) => self.a / self.b,
        }
    }

    fn variance(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => 0.0,
            (_, _) => self.a / (self.b * self.b),
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => 0.0,
            (_, _) => {
                self.a - self.b.ln() + gamma::ln_gamma(self.a) +
                (1.0 - self.a) * gamma::digamma(self.a)
            }
        }
    }

    fn skewness(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => 0.0,
            (_, _) => 2.0 / self.a.sqrt(),
        }
    }

    fn median(&self) -> Option<f64> {
        None
    }

    fn cdf(&self, x: f64) -> result::Result<f64> {
        match (self.a, self.b) {
            (0.0, 0.0) => Ok(0.0),
            (_, f64::INFINITY) => {
                if x == self.a {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            }
            (_, _) => gamma::gamma_lr(self.a, x * self.b),
        }
    }
}

impl Continuous for Gamma {
    fn mode(&self) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NAN,
            (_, f64::INFINITY) => self.a,
            (_, _) => (self.a - 1.0) / self.b,
        }
    }

    fn min(&self) -> f64 {
        0.0
    }

    fn max(&self) -> f64 {
        f64::INFINITY
    }

    fn pdf(&self, x: f64) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => 0.0,
            (_, f64::INFINITY) => {
                if x == self.a {
                    f64::INFINITY
                } else {
                    0.0
                }
            }
            (1.0, _) => self.b * (-self.b * x).exp(),
            (_, _) if self.a > 160.0 => self.ln_pdf(x).exp(),
            (_, _) => {
                self.b.powf(self.a) * x.powf(self.a - 1.0) * (-self.b * x).exp() /
                gamma::gamma(self.a)
            }
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        match (self.a, self.b) {
            (0.0, 0.0) => f64::NEG_INFINITY,
            (_, f64::INFINITY) => {
                if x == self.a {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            }
            (1.0, _) => self.b.ln() - self.b * x,
            (_, _) => {
                self.a * self.b.ln() + (self.a - 1.0) * x.ln() - self.b * x -
                gamma::ln_gamma(self.a)
            }
        }
    }
}

fn sample_unchecked<R: Rng>(r: &mut R, shape: f64, rate: f64) -> f64 {
    if rate == f64::INFINITY {
        return shape;
    }

    let mut a = shape;
    let mut afix = 1.0;

    if shape < 1.0 {
        a = shape + 1.0;
        afix = r.next_f64().powf(1.0 / shape);
    }

    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x = normal::sample_unchecked(r, 0.0, 1.0);
        let mut v = 1.0 + c * x;
        while v <= 0.0 {
            x = normal::sample_unchecked(r, 0.0, 1.0);
            v = 1.0 + c * x;
        }

        v = v * v * v;
        x = x * x;
        let u = r.next_f64();
        if u < 1.0 - 0.0331 * x * x {
            return afix * d * v / rate;
        }
        if u.ln() < 0.5 * x + d * (1.0 - v - v.ln()) {
            return afix * d * v / rate;
        }
    }
}

#[cfg(test)]
mod test {
    use std::f64;
    use std::option::Option;
    use distribution::{Univariate, Continuous};
    use prec;
    use result;
    use super::Gamma;
    
    fn try_create(shape: f64, rate: f64) -> Gamma {
        let n = Gamma::new(shape, rate);
        assert!(n.is_ok());
        n.unwrap()
    }
    
    fn create_case(shape: f64, rate: f64) {
        let n = try_create(shape, rate);
        assert_eq!(shape, n.shape());
        assert_eq!(rate, n.rate());
    }
    
    fn bad_create_case(shape: f64, rate: f64) {
        let n = Gamma::new(shape, rate);
        assert!(n.is_err());
    }
    
    fn test_case<F>(shape: f64, rate: f64, expected: f64, eval: F)
        where F : Fn(Gamma) -> f64 {
    
        let n = try_create(shape, rate);
        let x = eval(n);
        assert_eq!(expected, x);  
    }
    
    fn test_almost<F>(shape: f64, rate: f64, expected: f64, acc: f64, eval: F)
        where F : Fn(Gamma) -> f64 {
        
        let n = try_create(shape, rate);
        let x = eval(n);
        assert!(prec::almost_eq(expected, x, acc)); 
    }
    
    fn test_unsupported<F>(shape: f64, rate: f64, eval: F)
        where F : Fn(Gamma) -> Option<f64> {
    
        let n = try_create(shape, rate);
        let x = eval(n);
        assert!(x.is_none());        
    }
    
    fn test_result<F>(shape: f64, rate: f64, expected: f64, eval: F)
        where F : Fn(Gamma) -> result::Result<f64> {
        
        let n = try_create(shape, rate);
        let x = eval(n);
        assert!(x.is_ok());     
        
        let v = x.unwrap();
        assert_eq!(expected, v);
    }
    
    fn test_result_almost<F>(shape: f64, rate: f64, expected: f64, acc: f64, eval: F)
        where F : Fn(Gamma) -> result::Result<f64> {
        
        let n = try_create(shape, rate);
        let x = eval(n);
        assert!(x.is_ok());     
        
        let v = x.unwrap();
        assert!(prec::almost_eq(expected, v, acc));    
    }
    
    #[test]
    fn test_create() {
        create_case(0.0, 0.0);
        create_case(1.0, 0.1);
        create_case(1.0, 1.0);
        create_case(10.0, 10.0);
        create_case(10.0, 1.0);
        create_case(10.0, f64::INFINITY);
    }
    
    #[test]
    fn test_bad_create() {
        bad_create_case(1.0, f64::NAN);
        bad_create_case(1.0, -1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-1.0, -1.0);
        bad_create_case(-1.0, f64::NAN);
    }
    
    #[test]
    fn test_mean() {
        test_case(1.0, 0.1, 10.0, |x| x.mean());
        test_case(1.0, 1.0, 1.0, |x| x.mean());
        test_case(10.0, 10.0, 1.0, |x| x.mean());
        test_case(10.0, 1.0, 10.0, |x| x.mean());
        test_case(10.0, f64::INFINITY, 10.0, |x| x.mean());
    }
    
    #[test]
    fn test_variance() {
        test_almost(1.0, 0.1, 100.0, 1e-13, |x| x.variance());
        test_case(1.0, 1.0, 1.0, |x| x.variance());
        test_case(10.0, 10.0, 0.1, |x| x.variance());
        test_case(10.0, 1.0, 10.0, |x| x.variance());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.variance());
    }
    
    #[test]
    fn test_std_dev() {
        test_case(1.0, 0.1, 10.0, |x| x.std_dev());
        test_case(1.0, 1.0, 1.0, |x| x.std_dev());
        test_case(10.0, 10.0, 0.31622776601683794197697302588502426416723164097476643, |x| x.std_dev());
        test_case(10.0, 1.0, 3.1622776601683793319988935444327185337195551393252168, |x| x.std_dev());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.std_dev());
    }
    
    #[test]
    fn test_entropy() {
        test_almost(1.0, 0.1, 3.3025850929940456285068402234265387271634735938763824, 1e-15, |x| x.entropy());
        test_almost(1.0, 1.0, 1.0, 1e-15, |x| x.entropy());
        test_almost(10.0, 10.0, 0.23346908548693395836262094490967812177376750477943892, 1e-13, |x| x.entropy());
        test_almost(10.0, 1.0, 2.5360541784809796423806123995940423293748689934081866, 1e-13, |x| x.entropy());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.entropy());
    }
    
    #[test]
    fn test_skewness() {
        test_case(1.0, 0.1, 2.0, |x| x.skewness());
        test_case(1.0, 1.0, 2.0, |x| x.skewness());
        test_case(10.0, 10.0, 0.63245553203367586639977870888654370674391102786504337, |x| x.skewness());
        test_case(10.0, 1.0, 0.63245553203367586639977870888654370674391102786504337, |x| x.skewness());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.skewness());
    }
    
    #[test]
    fn test_mode() {
        test_case(1.0, 0.1, 0.0, |x| x.mode());
        test_case(1.0, 1.0, 0.0, |x| x.mode());
        test_case(10.0, 10.0, 0.9, |x| x.mode());
        test_case(10.0, 1.0, 9.0, |x| x.mode());
        test_case(10.0, f64::INFINITY, 10.0, |x| x.mode());
    }
    
    #[test]
    fn test_median() {
        test_unsupported(1.0, 0.1, |x| x.median());
        test_unsupported(1.0, 1.0, |x| x.median());
        test_unsupported(10.0, 10.0, |x| x.median());
        test_unsupported(10.0, 1.0, |x| x.median());
        test_unsupported(10.0, f64::INFINITY, |x| x.median());
    }
    
    #[test]
    fn test_min_max() {
        test_case(1.0, 0.1, 0.0, |x| x.min());
        test_case(1.0, 1.0, 0.0, |x| x.min());
        test_case(10.0, 10.0, 0.0, |x| x.min());
        test_case(10.0, 1.0, 0.0, |x| x.min());
        test_case(10.0, f64::INFINITY, 0.0, |x| x.min());
        test_case(1.0, 0.1, f64::INFINITY, |x| x.max());
        test_case(1.0, 1.0, f64::INFINITY, |x| x.max());
        test_case(10.0, 10.0, f64::INFINITY, |x| x.max());
        test_case(10.0, 1.0, f64::INFINITY, |x| x.max());
        test_case(10.0, f64::INFINITY, f64::INFINITY, |x| x.max());
    }
    
    #[test]
    fn test_pdf() {
        test_case(1.0, 0.1, 0.10000000000000000555111512312578270211815834045410156, |x| x.pdf(0.0));
        test_case(1.0, 0.1, 0.090483741803595961836995913651194571475319347018875963, |x| x.pdf(1.0));
        test_case(1.0, 0.1, 0.036787944117144234201693506390001264039984687455876246, |x| x.pdf(10.0));
        test_case(1.0, 1.0, 1.0, |x| x.pdf(0.0));
        test_case(1.0, 1.0, 0.36787944117144232159552377016146086744581113103176804, |x| x.pdf(1.0));
        test_case(1.0, 1.0, 0.000045399929762484851535591515560550610237918088866564953, |x| x.pdf(10.0));
        test_case(10.0, 10.0, 0.0, |x| x.pdf(0.0));
        test_almost(10.0, 10.0, 1.2511003572113329898476497894772544708420990097708588, 1e-14, |x| x.pdf(1.0));
        test_almost(10.0, 10.0, 1.0251532120868705806216092933926141802686541811003037e-30, 1e-44, |x| x.pdf(10.0));
        test_case(10.0, 1.0, 0.0, |x| x.pdf(0.0));
        test_almost(10.0, 1.0, 0.0000010137771196302974029859010421116095333052555418644397, 1e-20, |x| x.pdf(1.0));
        test_almost(10.0, 1.0, 0.12511003572113329898476497894772544708420990097708601, 1e-15, |x| x.pdf(10.0));
        test_case(10.0, f64::INFINITY, 0.0, |x| x.pdf(0.0));
        test_case(10.0, f64::INFINITY, 0.0, |x| x.pdf(1.0));
        test_case(10.0, f64::INFINITY, f64::INFINITY, |x| x.pdf(10.0));
    }
    
    #[test]
    fn test_ln_pdf() {
        test_case(1.0, 0.1, -2.3025850929940456285068402234265387271634735938763824, |x| x.ln_pdf(0.0));
        test_case(1.0, 0.1, -2.402585092994045634057955346552321429281631934330484, |x| x.ln_pdf(1.0));
        test_case(1.0, 0.1, -3.3025850929940456285068402234265387271634735938763824, |x| x.ln_pdf(10.0));
        test_case(1.0, 1.0, 0.0, |x| x.ln_pdf(0.0));
        test_case(1.0, 1.0, -1.0, |x| x.ln_pdf(1.0));
        test_case(1.0, 1.0, -10.0, |x| x.ln_pdf(10.0));
        test_case(10.0, 10.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_almost(10.0, 10.0, 0.22402344985898722897219667227693591172986563062456522, 1e-15, |x| x.ln_pdf(1.0));
        test_case(10.0, 10.0, -69.052710713194601614865880235563786219860220971716511, |x| x.ln_pdf(10.0));
        test_case(10.0, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_almost(10.0, 1.0, -13.801827480081469611207717874566706164281149255663166, 1e-14, |x| x.ln_pdf(1.0));
        test_almost(10.0, 1.0, -2.0785616431350584550457947824074282958712358580042068, 1e-14, |x| x.ln_pdf(10.0));
        test_case(10.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_case(10.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
        test_case(10.0, f64::INFINITY, f64::INFINITY, |x| x.ln_pdf(10.0));
    }
    
    #[test]
    fn test_cdf() {
        test_result(1.0, 0.1, 0.0, |x| x.cdf(0.0));
        test_result_almost(1.0, 0.1, 0.095162581964040431858607615783064404690935346242622848, 1e-16, |x| x.cdf(1.0));
        test_result_almost(1.0, 0.1, 0.63212055882855767840447622983853913255418886896823196, 1e-15, |x| x.cdf(10.0));
        test_result(1.0, 1.0, 0.0, |x| x.cdf(0.0));
        test_result_almost(1.0, 1.0, 0.63212055882855767840447622983853913255418886896823196, 1e-15, |x| x.cdf(1.0));
        test_result(1.0, 1.0, 0.99995460007023751514846440848443944938976208191113396, |x| x.cdf(10.0));
        test_result(10.0, 10.0, 0.0, |x| x.cdf(0.0));
        test_result_almost(10.0, 10.0, 0.54207028552814779168583514294066541824736464003242184, 1e-15, |x| x.cdf(1.0));
        test_result(10.0, 10.0, 0.99999999999999999999999999999988746526039157266114706, |x| x.cdf(10.0));
        test_result(10.0, 1.0, 0.0, |x| x.cdf(0.0));
        test_result_almost(10.0, 1.0, 0.00000011142547833872067735305068724025236288094949815466035, 1e-21, |x| x.cdf(1.0));
        test_result_almost(10.0, 1.0, 0.54207028552814779168583514294066541824736464003242184, 1e-15, |x| x.cdf(10.0));
        test_result(10.0, f64::INFINITY, 0.0, |x| x.cdf(0.0));
        test_result(10.0, f64::INFINITY, 0.0, |x| x.cdf(1.0));
        test_result(10.0, f64::INFINITY, 1.0, |x| x.cdf(10.0));
    }
}
