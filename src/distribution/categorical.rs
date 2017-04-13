/// Implements the [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution)
/// distribution, also known as the generalized Bernoulli or discrete distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Categorical, Discrete};
/// use statrs::statistics::Mode;
///
/// let n = Cauchy::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.1591549430918953357689);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Categorical {

}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use distribution::Categorical;
}