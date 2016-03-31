#![crate_type = "lib"]
#![crate_name = "statrs"]

extern crate rand;

pub mod distribution;
pub mod function;
pub mod consts;
pub mod prec;

mod result;
mod error;

pub use result::Result;
pub use error::StatsError;
