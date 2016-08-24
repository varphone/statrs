//! This crate aims to be a functional
//! port of the Math.NET Numerics Distribution package and in doing so providing the Rust numerical
//! computing community with a robust, well-tested statistical distribution package. This crate
//! also ports over some of the special statistical functions from Math.NET in so far as they are
//! used in the computation of distribution values. This crate depends on the `rand` crate to provide
//! RNG.
//!
//! # Example
//! The following example samples from a standard normal distribution
//!
//! ```
//! # extern crate rand;
//! # extern crate statrs;

//! use rand::StdRng;
//! use statrs::distribution::{Distribution, Normal};
//!
//! # fn main() {
//! let mut r = rand::StdRng::new().unwrap();
//! let n = Normal::new(0.0, 1.0).unwrap();
//! for _ in 0..10 {
//!     print!("{}", n.sample::<StdRng>(&mut r));
//! }
//! # }
//! ```

#![crate_type = "lib"]
#![crate_name = "statrs"]

extern crate rand;

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $prec:expr) => (
        if !$crate::prec::almost_eq($a, $b, $prec) {
            panic!(format!("assertion failed: `abs(left - right) < {:e}`, (left: `{}`, right: `{}`)", $prec, $a, $b));
        } 
    );
}

pub mod distribution;
pub mod function;
pub mod consts;
pub mod prec;

mod result;
mod statistics;
mod error;

pub use result::Result;
pub use statistics::Statistics;
pub use error::StatsError;
