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
