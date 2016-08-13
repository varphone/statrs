# statrs  
  
[![Build Status](https://travis-ci.org/boxtown/statrs.svg?branch=master)](https://travis-ci.org/boxtown/verto)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Crates.io](https://img.shields.io/crates/v/statrs.svg?maxAge=2592000)]()  

## Current Version: v0.1.0

Should be stable for both nightly and stable Rust

## Description
  
A work-in-progress port of the Math.NET Numerics distributions package to Rust.  
All unit tests are borrowed from the Math.NET source when possible and filled in  
when not. 

Please check out the documentation [here](https://boxtown.io/docs/statrs/0.1.0/statrs/)

## Usage

Add the following to your `Cargo.toml`

```Rust
[dependencies]
statrs = "0.1.0"
```

and this to your crate root

```Rust
extern crate statrs;
```
  
## Examples

Statrs v0.1.0 comes with a number of commonly used distributions including Normal, Gamma, Student's T, Exponential, Weibull, etc.
The common use case is to set up the distributions and sample from them which depends on the `Rand` crate for random number generation

```Rust
use rand;
use statrs::distribution::{Exponential, Distribution};

let mut r = rand::StdRng::new().unwrap();
let n = Exponential::new(0.5).unwrap();
print!("{}", n.Sample::<StdRng>(&mut r);
```

Statrs also comes with a number of useful utility traits for more detailed introspection of distributions

```Rust
use statrs::distribution::{Exponential, Mean, Variance, Entropy, Skewness, Univariate, Continuous};

let n = Exponential::new(1.0).unwrap();
assert_eq!(n.mean(), 1.0);
assert_eq!(n.variance(), 1.0);
assert_eq!(n.entropy(), 1.0);
assert_eq!(n.skewness(), 2.0);
assert_eq!(n.cdf(1.0), 0.6321205588285576784045);
assert_eq!(n.pdf(1.0), 0.3678794411714423215955);
```

as well as utility functions including `erf`, `gamma`, `ln_gamma`, `beta`, etc
