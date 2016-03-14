#![crate_type = "lib"]
#![crate_name = "statrs"]

extern crate rand;

pub use result::Result;
pub use error::StatsError;

pub mod distribution;
pub mod functions;
pub mod consts;
pub mod prec;

mod result;
mod error;
