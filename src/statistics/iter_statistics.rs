use std::f64;
use std::borrow::Borrow;
use error::StatsError;
use statistics::*;

impl<T> Statistics<f64> for T
    where T: IntoIterator,
          T::Item: Borrow<f64>
{
    fn abs_min(self) -> f64 {
        let mut iter = self.into_iter();
        match iter.next() {
            None => f64::NAN,
            Some(init) => {
                iter.map(|x| x.borrow().abs())
                    .fold(init.borrow().abs(),
                          |acc, x| if x < acc || x.is_nan() { x } else { acc })
            }
        }
    }

    fn abs_max(self) -> f64 {
        let mut iter = self.into_iter();
        match iter.next() {
            None => f64::NAN,
            Some(init) => {
                iter.map(|x| x.borrow().abs())
                    .fold(init.borrow().abs(),
                          |acc, x| if x > acc || x.is_nan() { x } else { acc })
            }
        }
    }

    fn geometric_mean(self) -> f64 {
        let mut i = 0.0;
        let mut sum = 0.0;
        for x in self {
            i += 1.0;
            sum += x.borrow().ln();
        }
        if i > 0.0 { (sum / i).exp() } else { f64::NAN }
    }

    fn harmonic_mean(self) -> f64 {
        let mut i = 0.0;
        let mut sum = 0.0;
        for x in self {
            i += 1.0;

            let borrow = *x.borrow();
            if borrow < 0f64 {
                return f64::NAN;
            }
            sum += 1.0 / borrow;
        }
        if i > 0.0 { i / sum } else { f64::NAN }
    }

    fn population_variance(self) -> f64 {
        let mut iter = self.into_iter();
        let mut sum = match iter.next() {
            None => return f64::NAN,
            Some(x) => *x.borrow(),
        };
        let mut i = 1.0;
        let mut variance = 0.0;

        for x in iter {
            i += 1.0;
            let borrow = *x.borrow();
            sum += borrow;
            let diff = i * borrow - sum;
            variance += diff * diff / (i * (i - 1.0));
        }
        variance / i
    }

    fn population_std_dev(self) -> f64 {
        self.population_variance().sqrt()
    }

    fn covariance(self, other: Self) -> f64 {
        let mut n = 0.0;
        let mut mean1 = 0.0;
        let mut mean2 = 0.0;
        let mut comoment = 0.0;

        let mut iter = other.into_iter();
        for x in self {
            let borrow = *x.borrow();
            let borrow2 = match iter.next() {
                None => panic!(format!("{}", StatsError::ContainersMustBeSameLength)),
                Some(x) => *x.borrow(),
            };
            let old_mean2 = mean2;
            n += 1.0;
            mean1 += (borrow - mean1) / n;
            mean2 += (borrow2 - mean2) / n;
            comoment += (borrow - mean1) * (borrow2 - old_mean2);
        }
        if iter.next().is_some() {
            panic!(format!("{}", StatsError::ContainersMustBeSameLength));
        }

        if n > 1.0 {
            comoment / (n - 1.0)
        } else {
            f64::NAN
        }
    }

    fn population_covariance(self, other: Self) -> f64 {
        let mut n = 0.0;
        let mut mean1 = 0.0;
        let mut mean2 = 0.0;
        let mut comoment = 0.0;

        let mut iter = other.into_iter();
        for x in self {
            let borrow = *x.borrow();
            let borrow2 = match iter.next() {
                None => panic!(format!("{}", StatsError::ContainersMustBeSameLength)),
                Some(x) => *x.borrow(),
            };
            let old_mean2 = mean2;
            n += 1.0;
            mean1 += (borrow - mean1) / n;
            mean2 += (borrow2 - mean2) / n;
            comoment += (borrow - mean1) * (borrow2 - old_mean2);
        }
        if iter.next().is_some() {
            panic!(format!("{}", StatsError::ContainersMustBeSameLength));
        }
        if n > 0.0 { comoment / n } else { f64::NAN }
    }

    fn quadratic_mean(self) -> f64 {
        let mut mean = 0.0;
        let mut count = 0.0;
        for x in self {
            let borrow = *x.borrow();
            count += 1.0;
            mean += (borrow * borrow - mean) / count;
        }
        if count > 0.0 { mean.sqrt() } else { f64::NAN }
    }
}

// fn mean(self) -> f64 {
//     let mut i = 0.0;
//     let mut mean = 0.0;
//     for x in self {
//         i += 1.0;
//         mean += (x.borrow() - mean) / i;
//     }
//     if i > 0.0 { mean } else { f64::NAN }
// }

// fn variance(self) -> f64 {
//     let iter = self.into_iter();
//     let mut sum = match iter.next() {
//         None => return f64::NAN,
//         Some(x) => *x.borrow(),
//     };
//     let mut i = 1.0;
//     let mut variance = 0.0;

//     for x in iter {
//         i += 1.0;
//         let borrow = *x.borrow();
//         sum += borrow;
//         let diff = i * borrow - sum;
//         variance += diff * diff / (i * (i - 1.0));
//     }
//     if i > 1.0 {
//         variance / (i - 1.0)
//     } else {
//         f64::NAN
//     }
// }

// fn std_dev(self) -> f64 {
//     self.variance().sqrt()
// }

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use statistics::IterStatistics;
    use testing;

    #[test]
    fn test_mean() {
        let mut data = testing::load_data("nist/lottery.txt");
        assert_almost_eq!(data.iter().mean(), 518.958715596330, 1e-12);

        data = testing::load_data("nist/lew.txt");
        assert_almost_eq!(data.iter().mean(), -177.435000000000, 1e-13);

        data = testing::load_data("nist/mavro.txt");
        assert_almost_eq!(data.iter().mean(), 2.00185600000000, 1e-15);

        data = testing::load_data("nist/michaelso.txt");
        assert_almost_eq!(data.iter().mean(), 299.852400000000, 1e-13);

        data = testing::load_data("nist/numacc1.txt");
        assert_eq!(data.iter().mean(), 10000002.0);

        data = testing::load_data("nist/numacc2.txt");
        assert_almost_eq!(data.iter().mean(), 1.2, 1e-15);

        data = testing::load_data("nist/numacc3.txt");
        assert_eq!(data.iter().mean(), 1000000.2);

        data = testing::load_data("nist/numacc4.txt");
        assert_almost_eq!(data.iter().mean(), 10000000.2, 1e-8);
    }
}
