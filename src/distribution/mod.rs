use rand::Rng;

pub use self::bernoulli::Bernoulli;
pub use self::binomial::Binomial;
pub use self::chi::Chi;
pub use self::chi_squared::ChiSquared;
pub use self::discrete_uniform::DiscreteUniform;
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::log_normal::LogNormal;
pub use self::normal::Normal;
pub use self::poisson::Poisson;
pub use self::triangular::Triangular;
pub use self::uniform::Uniform;

mod bernoulli;
mod binomial;
mod chi;
mod chi_squared;
mod discrete_uniform;
mod exponential;
mod gamma;
mod log_normal;
mod normal;
mod poisson;
mod triangular;
mod uniform;

/// Distribution is trait that should be implemented
/// by structs that represent a statistical distribution
pub trait Distribution {
    /// Draws a random sample according to the distribution
    /// and the supplied random number generator
    fn sample<R: Rng>(&self, r: &mut R) -> f64;
}

/// Univariate should be implemented by structs
/// representing a univariate statistical distribution.
pub trait Univariate : Distribution {
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn std_dev(&self) -> f64;
    fn entropy(&self) -> f64;
    fn skewness(&self) -> f64;
    fn median(&self) -> f64;
    fn cdf(&self, x: f64) -> f64;
}

/// Continuous should be implemented by structs
/// representing a continuous univariate statistical
/// distribution
pub trait Continuous : Univariate {
    fn mode(&self) -> f64;
    fn min(&self) -> f64;
    fn max(&self) -> f64;
    fn pdf(&self, x: f64) -> f64;
    fn ln_pdf(&self, x: f64) -> f64;
}

/// Discrete should be implemented by structs
/// representing a discrete univariate statistical
/// distribution
pub trait Discrete : Univariate {
    fn mode(&self) -> i64;
    fn min(&self) -> i64;
    fn max(&self) -> i64;
    fn pmf(&self, x: i64) -> f64;
    fn ln_pmf(&self, x: i64) -> f64;
}
