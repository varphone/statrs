//! Provides number theory utility functions

pub trait Modulus {
    fn modulus(self, divisor: Self) -> Self;
}

impl Modulus for f64 {
    fn modulus(self, divisor: f64) -> f64 {
        ((self % divisor) + divisor) % divisor
    }
}

impl Modulus for f32 {
    fn modulus(self, divisor: f32) -> f32 {
        ((self % divisor) + divisor) % divisor
    }
}

impl Modulus for i64 {
    fn modulus(self, divisor: i64) -> i64 {
        ((self % divisor) + divisor) % divisor
    }
}

impl Modulus for i32 {
    fn modulus(self, divisor: i32) -> i32 {
        ((self % divisor) + divisor) % divisor
    }
}

impl Modulus for u64 {
    fn modulus(self, divisor: u64) -> u64 {
        ((self % divisor) + divisor) % divisor
    }
}

impl Modulus for u32 {
    fn modulus(self, divisor: u32) -> u32 {
        ((self % divisor) + divisor) % divisor
    }
}
