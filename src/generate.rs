//! Provides utility functions for generating data sequences
use std::f64::consts;
use euclid::Modulus;

/// Generates a linearly spaced vector of the given length between
/// the specified values (inclusive). Equivalent to MATLAB linspace
///
/// # Note
///
/// Will become deprecated once `step_by` becomes stable
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::linear_spaced(9, 0.0, 64.0);
/// assert_eq!(x, [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0]);
/// ```
pub fn linear_spaced(length: usize, start: f64, stop: f64) -> Vec<f64> {
    match length {
        0 => Vec::new(),
        1 => vec![stop],
        _ => {
            let step = (stop - start) / (length - 1) as f64;
            let mut vec = (0..length).map(|x| start + (x as f64) * step).collect::<Vec<f64>>();
            vec[length - 1] = stop;
            vec
        }
    }
}

/// Creates a vector of `f64` points representing a periodic wave with an amplitude
/// of `1.0`, phase of `0.0`, and delay of `0`.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::periodic(10, 8.0, 2.0);
/// assert_eq!(x, [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25]);
/// ```
pub fn periodic(length: usize, sampling_rate: f64, frequency: f64) -> Vec<f64> {
    periodic_custom(length, sampling_rate, frequency, 1.0, 0.0, 0)
}

/// Creates a vector of `f64` points representing a periodic wave.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::periodic_custom(10, 8.0, 2.0, 10.0, 1.0, 2);
/// assert_eq!(x, [6.0, 8.5, 1.0, 3.5, 6.0, 8.5, 1.0, 3.5, 6.0, 8.5]);
/// ```
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

    for d in &mut data {
        let mut x = phase + k * step;
        if x >= amplitude {
            x %= amplitude;
            phase = x;
            k = 0.0;
        }
        *d = x;
        k += 1.0;
    }
    data
}

/// Creates a vector of `f64` points representing a Sine wave with a mean of `0.0`,
/// phase of `0.0`, and delay of `0`.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::sinusoidal(10, 8.0, 2.0, 1.0);
/// assert_eq!(x,
///     [0.0, 1.0, 0.00000000000000012246467991473532,
///     -1.0, -0.00000000000000024492935982947064, 1.0,
///     0.00000000000000036739403974420594, -1.0,
///     -0.0000000000000004898587196589413, 1.0]);
/// ```
pub fn sinusoidal(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64) -> Vec<f64> {
    sinusoidal_custom(length, sampling_rate, frequency, amplitude, 0.0, 0.0, 0)
}

/// Creates a vector of `f64` points representing a Sine wave.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::sinusoidal_custom(10, 8.0, 2.0, 1.0, 5.0, 2.0, 1);
/// assert_eq!(x,
///     [5.416146836547142, 5.909297426825682, 4.583853163452858,
///     4.090702573174318, 5.416146836547142, 5.909297426825682,
///     4.583853163452858, 4.090702573174318, 5.416146836547142,
///     5.909297426825682]);
/// ```
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
    (0..length)
        .map(|i| mean + amplitude * (phase + i as f64 * step).sin())
        .collect()
}
