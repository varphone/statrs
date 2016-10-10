use std::iter::Iterator;
use statistics::*;

impl Statistics for Iterator<Item=f64> {
    fn abs_min(&self) -> f64 {
        unimplemented!()
    }

    fn abs_max(&self) -> f64 {
        unimplemented!()
    }

    fn geometric_mean(&self) -> f64 {
        unimplemented!()
    }

    fn harmonic_mean(&self) -> f64 {
        unimplemented!()
    }

    fn population_variance(&self) -> f64 {
        unimplemented!()
    }

    fn population_std_dev(&self) -> f64 {
        unimplemented!()
    }

    fn covariance(&self, other: &Self) -> f64 {
        unimplemented!()
    }

    fn population_covariance(&self, other: &Self) -> f64 {
        unimplemented!()
    }

    fn quadratic_mean(&self) -> f64 {
        unimplemented!()
    }

    fn order_statistic(&self, order: usize) -> f64 {
        unimplemented!()
    }

    fn quantile(&self, tau: f64) -> f64 {
        unimplemented!()
    }

    fn percentile(&self, p: usize) -> f64 {
        unimplemented!()
    }

    fn lower_quartile(&self) -> f64 {
        unimplemented!()
    }

    fn upper_quartile(&self) -> f64 {
        unimplemented!()
    }

    fn interquartile_range(&self) -> f64 {
        unimplemented!()
    }

    fn ranks(&self, tie_breaker: RankTieBreaker) -> Vec<f64> {
        unimplemented!()
    }
}