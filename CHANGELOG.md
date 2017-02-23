v0.6.0
 - `gamma::gamma_ur`, `gamma::gamma_ui`, `gamma::gamma_lr`, and `gamma::gamma_li` now follow strict gamma function domain, panicking if `a` or `x` are not in `(0, +inf)`
 - `beta::beta_reg` no longer allows `0.0` for `a` or `b` arguments
 - `InverseGamma` no longer accepts `f64::INFINITY` as valid arguments for `shape` or `rate` as the value is nonsense

v0.5.1
 - Fixed critical bug in `normal::sample_unchecked` where it was returning `NaN`

v0.5.0
 - Implemented the `logistic::logistic` special function
 - Implemented the `logistic::logit` special function
 - Implemented the `factorial::multinomial` special function
 - Implemented the `harmonic::harmonic` special function
 - Implemented the `harmonic::gen_harmonic` special function
 - Implemented the `InverseGamma` distribution
 - Implemented the `Geometric` distribution
 - Implemented the `Hypergeometric ` distribution
 - `gamma::gamma_ur` now panics when `x > 0` or `a == f64::NEG_INFINITY`. In addition, it also returns `f64::NAN` when `a == f64::INFINITY` and `0.0` when `x == f64::INFINITY`
 - `Gamma::pdf` and `Gamma::ln_pdf` now return `f64::NAN` if any of `shape`, `rate`, or `x` are `f64::INFINITY`
 - `Binomial::pdf` and `Binomial::ln_pdf` now panic if `x > n` or `x < 0`
 - `Bernoulli::pdf` and `Bernoulli::ln_pdf` now panic if `x > 1` or `x < 0`

v0.4.0
- Implemented the `exponential::integral` special function
- Implemented the `Cauchy` (otherwise known as the `Lorenz`) distribution
- Implemented the `Dirichlet` distribution
- `Continuous` and `Discrete` traits no longer dependent on `Distribution` trait

v0.3.2
- Implemented the `FisherSnedecor` (F) distribution

v0.3.1
- Removed print statements from `ln_pdf` method in `Beta` distribution

v0.3.0
- Moved methods `min` and `max` out of trait `Univariate` into their own respective traits `Min` and `Max`
- Traits `Min`, `Max`, `Mean`, `Variance`, `Entropy`, `Skewness`, `Median`, and `Mode` moved from `distribution` module to `statistics` module
- `Mean`, `Variance`, `Entropy`, `Skewness`, `Median`, and `Mode` no longer depend on `Distribution` trait
- `Mean`, `Variance`, `Skewness`, and `Mode` are now generic over only one type, the return type, due to not depending on `Distribution` anymore
- `order_statistic`, `median`, `quantile`, `percentile`, `lower_quartile`, `upper_quartile`, `interquartile_range`, and `ranks` methods removed
    from `Statistics` trait. 
- `min`, `max`, `mean`, `variance`, and `std_dev` methods added to `Statistics` trait
- `Statistics` trait now implemented for all types implementing `IntoIterator` where `Item` implements `Borrow<f64>`. Slice now implicitly implements
    `Statistics` through this new implementation.
- Slice still implements `Min`, `Max`, `Mean`, and `Variance` but now through the `Statistics` implementation rather than its own implementation
- `InplaceStatistics` renamed to `OrderStatistics`, all methods in `InplaceStatistics` have `_inplace` trimmed from method name.
- Inverse DiGamma function implemented with signature `gamma::inv_digamma(x: f64) -> f64`

v0.2.0
- Created `statistics` module and `Statistics` trait
- `Statistics` trait implementation for `[f64]`
- Implemented `Beta` distribution
- Added `Modulus` trait and implementations for `f32`, `f64`, `i32`, `i64`, `u32`, and `u64` in `euclid` module
- Added periodic and sinusoidal vector generation functions in `generate` module
