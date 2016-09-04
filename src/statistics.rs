use std::f64;
use error::StatsError;

pub trait Statistics {
    /// Returns the minimum value in the data
    ///
    /// # Rermarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn min(&self) -> f64;

    /// Returns the maximum value in the data
    ///
    /// # Rermarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn max(&self) -> f64;

    /// Returns the minimum absolute value in the data
    ///
    /// # Rermarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn abs_min(&self) -> f64;

    /// Returns the maximum absolute value in the data
    ///
    /// # Rermarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn abs_max(&self) -> f64;

    /// Evaluates the sample mean, an estimate of the population
    /// mean.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn mean(&self) -> f64;

    /// Evaluates the geometric mean of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn geometric_mean(&self) -> f64;

    /// Evaluates the harmonic mean of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn harmonic_mean(&self) -> f64;

    /// Estimates the unbiased population variance from the provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is `f64::NAN`
    fn variance(&self) -> f64;

    /// Evaluates the population variance from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn population_variance(&self) -> f64;

    /// Estimates the unbiased population standard deviation from the provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is `f64::NAN`
    fn std_dev(&self) -> f64;

    /// Evaluates the population standard deviation from a full population.
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus
    /// be biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    fn population_std_dev(&self) -> f64;

    /// Estimates the unbiased population covariance between the two provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is `f64::NAN`
    ///
    /// # Panics
    ///
    /// If the two sample containers do not contain the same number of elements
    fn covariance(&self, other: &Self) -> f64;

    /// Evaluates the population covariance between the two provider populations
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N` is used as a normalizer and would thus be
    /// biased if applied to a subset
    ///
    /// Returns `f64::NAN` if data is empty or any entry is `f64::NAN`
    ///
    /// # Panics
    ///
    /// If the two sample containers do not contain the same number of elements
    fn population_covariance(&self, other: &Self) -> f64;

    /// Estimates the quadratic mean (Root Mean Square) of the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or any entry is `f64::NAN`
    fn quadratic_mean(&self) -> f64;

    /// Returns the order statistic `(order 1..N)` from the data
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Order must be one-based (between `1` and `N` inclusive)
    /// Returns `f64::NAN` if order is outside the viable range or data is empty.
    fn order_statistic(&mut self, order: usize) -> f64;

    /// Returns the median value from the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    fn median(&mut self) -> f64;

    /// Estimates the tau-th quantile from the data. The tau-th quantile
    /// is the data value where the cumulative distribution function crosses tau.
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Tau must be between `0` and `1` inclusive.
    /// Returns `f64::NAN` if data is empty or tau is outside the inclusive range.
    fn quantile(&mut self, tau: f64) -> f64;
}

impl Statistics for [f64] {
    fn min(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        self.iter().fold(f64::INFINITY, |acc, &x| if x < acc || x.is_nan() { x } else { acc })
    }

    fn max(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        self.iter().fold(f64::NEG_INFINITY, |acc, &x| if x > acc || x.is_nan() { x } else { acc })
    }

    fn abs_min(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        self.iter()
            .map(|x| x.abs())
            .fold(f64::INFINITY, |acc, x| if x < acc || x.is_nan() { x } else { acc })
    }

    fn abs_max(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        self.iter()
            .map(|x| x.abs())
            .fold(f64::NEG_INFINITY, |acc, x| if x > acc || x.is_nan() { x } else { acc })
    }

    fn mean(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        let mut m = 0.0;
        self.iter()
            .fold(0.0, |acc, &x| {
                m += 1.0;
                acc + (x - acc) / m
            })
    }

    fn geometric_mean(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        (self.iter().fold(0.0, |acc, &x| acc + x.ln()) / self.len() as f64).exp()
    }

    fn harmonic_mean(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        self.len() as f64 / self.iter().fold(0.0, |acc, &x| acc + 1.0 / x)
    }

    fn variance(&self) -> f64 {
        if self.len() <= 1 {
            return f64::NAN;
        }

        unsafe {
            let mut var = 0.0;
            let mut t = *self.get_unchecked(0);
            for i in 1..self.len() {
                t += *self.get_unchecked(i);
                let diff = (i as f64 + 1.0) * *self.get_unchecked(i) - t;
                var += (diff * diff) / ((i + 1) * i) as f64;
            }
            var / (self.len() - 1) as f64
        }
    }

    fn population_variance(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        unsafe {
            let mut var = 0.0;
            let mut t = *self.get_unchecked(0);
            for i in 1..self.len() {
                t += *self.get_unchecked(i);
                let diff = (i as f64 + 1.0) * *self.get_unchecked(i) - t;
                var += (diff * diff) / ((i + 1) * i) as f64
            }
            var / self.len() as f64
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn population_std_dev(&self) -> f64 {
        self.population_variance().sqrt()
    }

    fn covariance(&self, other: &[f64]) -> f64 {
        let n1 = self.len();
        let n2 = other.len();
        assert!(n1 == n2, format!("{}", StatsError::VectorsSameLength));
        if n1 <= 1 {
            return f64::NAN;
        }

        let mean1 = self.mean();
        let mean2 = other.mean();
        self.iter()
            .zip(other.iter())
            .fold(0.0, |acc, x| acc + (x.0 - mean1) * (x.1 - mean2)) / (n1 - 1) as f64
    }

    fn population_covariance(&self, other: &[f64]) -> f64 {
        let n1 = self.len();
        let n2 = other.len();
        assert!(n1 == n2, format!("{}", StatsError::VectorsSameLength));
        if n1 == 0 {
            return f64::NAN;
        }

        let mean1 = self.mean();
        let mean2 = other.mean();
        self.iter()
            .zip(other.iter())
            .fold(0.0, |acc, x| acc + (x.0 - mean1) * (x.1 - mean2)) / n1 as f64
    }

    fn quadratic_mean(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        let mut m = 0.0;
        self.iter()
            .fold(0.0, |acc, &x| {
                m += 1.0;
                acc + (x * x - acc) / m
            })
    }

    /// Returns the order statistic `(order 1..N)` from the data
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Order must be one-based (between `1` and `N` inclusive).
    /// Returns `f64::NAN` if order is outside the viable range or data is empty.
    ///
    /// **NOTE:** This method works inplace for arrays and may cause the array to be reordered
    fn order_statistic(&mut self, order: usize) -> f64 {
        let n = self.len();
        match order {
            1 => self.min(),
            _ if order == n => self.max(),
            _ if order < 1 || order > n => f64::NAN,
            _ => select_inplace(self, order - 1)
        }
    }

    /// Returns the median value from the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// **NOTE:** This method works inplace for arrays and may cause the array to be reordered
    fn median(&mut self) -> f64 {
        let k = self.len() / 2;
        if self.len() % 2 != 0 {
            select_inplace(self, k)
        } else {
            (select_inplace(self, k.saturating_sub(1)) + select_inplace(self, k)) / 2.0
        }
    }

    /// Estimates the tau-th quantile from the data. The tau-th quantile
    /// is the data value where the cumulative distribution function crosses tau.
    ///
    /// # Remarks
    ///
    /// No sorting is assumed. Tau must be between `0` and `1` inclusive.
    /// Returns `f64::NAN` if data is empty or tau is outside the inclusive range.
    ///
    /// **NOTE:** This method works inplace for arrays and may cause the array to be reordered
    fn quantile(&mut self, tau: f64) -> f64 {
        if tau < 0.0 || tau > 1.0 || self.len() == 0 {
            return f64::NAN;
        }

        let h = (self.len() as f64 + 1.0 / 3.0) * tau + 1.0 / 3.0;
        let hf = h as i64;

        if hf <= 0 || tau == 0.0 {
            return self.min();
        }
        if hf >= self.len() as i64 || tau == 1.0 {
            return self.max();
        }

        let a = select_inplace(self, (hf as usize).saturating_sub(1));
        let b = select_inplace(self, hf as usize);
        a + (h - hf as f64) * (b - a)
    }
}

// Selection algorithm from Numerical Recipes
// See: https://en.wikipedia.org/wiki/Selection_algorithm
fn select_inplace(arr: &mut [f64], rank: usize) -> f64 {
    if rank == 0 {
        return arr.min();
    }
    if rank > arr.len() - 1 {
        return arr.max();
    }

    unsafe {
        let mut low = 0;
        let mut high = arr.len() - 1;
        loop {
            if high <= low + 1 {
                if high == low + 1 && *arr.get_unchecked(high) < *arr.get_unchecked(low) {
                    arr.swap(low, high)
                }
                return *arr.get_unchecked(rank);
            }

            let middle = (low + high) >> 1;
            arr.swap(middle, low + 1);

            if *arr.get_unchecked(low) > *arr.get_unchecked(high) {
                arr.swap(low, high);
            }
            if *arr.get_unchecked(low + 1) > *arr.get_unchecked(high) {
                arr.swap(low + 1, high);
            }
            if *arr.get_unchecked(low) > *arr.get_unchecked(low + 1) {
                arr.swap(low, low + 1);
            }

            let mut begin = low + 1;
            let mut end = high;
            let pivot = *arr.get_unchecked(begin);
            loop {
                loop {
                    begin += 1;
                    if *arr.get_unchecked(begin) < pivot {
                        break;
                    }
                }
                loop {
                    end -= 1;
                    if *arr.get_unchecked(end) > pivot {
                        break;
                    }
                }
                if end < begin {
                    break;
                }
                arr.swap(begin, end);
            }

            arr[low + 1] = *arr.get_unchecked(end);
            arr[end] = pivot;

            if end >= rank {
                high = end - 1;
            }
            if end <= rank {
                low = begin;
            }
        }
    }
}
