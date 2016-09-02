use std::f64;
use error::StatsError;

pub trait Statistics {
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
}

impl Statistics for [f64] {
    fn mean(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        let mut m = 0.0;
        self.iter()
            .fold(0.0, |acc, x| {
                m += 1.0;
                acc + (x - acc) / m
            })
    }

    fn geometric_mean(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        (self.iter().fold(0.0, |acc, x| acc + x.ln()) / self.len() as f64).exp()
    }

    fn harmonic_mean(&self) -> f64 {
        if self.len() == 0 {
            return f64::NAN;
        }

        self.len() as f64 / self.iter().fold(0.0, |acc, x| acc + 1.0 / x)
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
}
