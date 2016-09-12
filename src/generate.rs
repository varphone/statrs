//! Provides utility functions for generating data sequences
use std::f64::consts;
use euclid::Modulus;

pub fn periodic(length: usize, sampling_rate: f64, frequency: f64) -> Vec<f64> {
    periodic_custom(length, sampling_rate, frequency, 1.0, 0.0, 0)
}

pub fn periodic_custom(length: usize,
                       sampling_rate: f64,
                       frequency: f64,
                       amplitude: f64,
                       phase: f64,
                       delay: i64)
                       -> Vec<f64> {

    let step = frequency / sampling_rate * amplitude;
    let mut phase = (phase - delay as f64 * step).modulus(amplitude);
    let mut data = vec![0.0; length];
    let mut k = 0.0;

    for i in 0..length {
        let mut x = phase + k * step;
        if x >= amplitude {
            x %= amplitude;
            phase = x;
            k = 0.0;
        }
        data[i] = x;
        k += 1.0;
    }
    data
}

pub fn sinusoidal(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64) -> Vec<f64> {
    sinusoidal_custom(length, sampling_rate, frequency, amplitude, 0.0, 0.0, 0)
}

pub fn sinusoidal_custom(length: usize,
                         sampling_rate: f64,
                         frequency: f64,
                         amplitude: f64,
                         mean: f64,
                         phase: f64,
                         delay: i64)
                         -> Vec<f64> {

    let pi2 = consts::PI * 2.0;
    let step = frequency / sampling_rate * pi2;
    let phase = (phase - delay as f64 * step) % pi2;
    let mut data = vec![0.0; length];

    for i in 0..length {
        data[i] = mean + amplitude * (phase + i as f64 * step).sin();
    }
    data
}
