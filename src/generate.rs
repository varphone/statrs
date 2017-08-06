//! Provides utility functions for generating data sequences
use std::f64::consts;
use euclid::Modulus;

/// Generates a base 10 log spaced vector of the given length between the specified
/// decade exponents (inclusive). Equivalent to MATLAB logspace
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::log_spaced(5, 0.0, 4.0);
/// assert_eq!(x, [1.0, 10.0, 100.0, 1000.0, 10000.0]);
/// ```
pub fn log_spaced(length: usize, start_exp: f64, stop_exp: f64) -> Vec<f64> {
    match length {
        0 => Vec::new(),
        1 => vec![10f64.powf(stop_exp)],
        _ => {
            let step = (stop_exp - start_exp) / (length - 1) as f64;
            let mut vec = (0..length)
                .map(|x| 10f64.powf(start_exp + (x as f64) * step))
                .collect::<Vec<f64>>();
            vec[length - 1] = 10f64.powf(stop_exp);
            vec
        }
    }
}

/// Finite iterator returning floats that form a periodic wave
pub struct Periodic {
    length: usize,
    amplitude: f64,
    step: f64,
    phase: f64,
    k: f64,
    i: usize,
}

impl Periodic {
    /// Constructs a new periodic wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Periodic;
    ///
    /// let x = Periodic::new(10, 8.0, 2.0, 10.0, 1.0, 2).collect::<Vec<f64>>();
    /// assert_eq!(x, [6.0, 8.5, 1.0, 3.5, 6.0, 8.5, 1.0, 3.5, 6.0, 8.5]);
    /// ```
    pub fn new(length: usize,
               sampling_rate: f64,
               frequency: f64,
               amplitude: f64,
               phase: f64,
               delay: i64)
               -> Periodic {

        let step = frequency / sampling_rate * amplitude;
        Periodic {
            length: length,
            amplitude: amplitude,
            step: step,
            phase: (phase - delay as f64 * step).modulus(amplitude),
            k: 0.0,
            i: 0,
        }
    }

    /// Constructs a default periodic wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Periodic;
    ///
    /// let x = Periodic::default(10, 8.0, 2.0).collect::<Vec<f64>>();
    /// assert_eq!(x, [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25]);
    /// ```
    pub fn default(length: usize, sampling_rate: f64, frequency: f64) -> Periodic {
        Self::new(length, sampling_rate, frequency, 1.0, 0.0, 0)
    }
}

impl Iterator for Periodic {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.i == self.length {
            None
        } else {
            let mut x = self.phase + self.k * self.step;
            if x >= self.amplitude {
                x %= self.amplitude;
                self.phase = x;
                self.k = 0.0;
            }
            self.k += 1.0;
            self.i += 1;
            Some(x)
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
#[deprecated(since="0.8.0", note="please use `Periodic::default` instead")]
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
#[deprecated(since="0.8.0", note="please use `Periodic::new` instead")]
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

/// Finite iterator returning floats that form a sinusoidal wave
pub struct Sinusoidal {
    length: usize,
    amplitude: f64,
    mean: f64,
    step: f64,
    phase: f64,
    i: usize,
}

impl Sinusoidal {
    /// Constructs a new sinusoidal wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Sinusoidal;
    ///
    /// let x = Sinusoidal::new(10, 8.0, 2.0, 1.0, 5.0, 2.0, 1).collect::<Vec<f64>>();
    /// assert_eq!(x,
    ///     [5.416146836547142, 5.909297426825682, 4.583853163452858,
    ///     4.090702573174318, 5.416146836547142, 5.909297426825682,
    ///     4.583853163452858, 4.090702573174318, 5.416146836547142,
    ///     5.909297426825682]);
    /// ```
    pub fn new(length: usize,
               sampling_rate: f64,
               frequency: f64,
               amplitude: f64,
               mean: f64,
               phase: f64,
               delay: i64)
               -> Sinusoidal {

        let pi2 = consts::PI * 2.0;
        let step = frequency / sampling_rate * pi2;
        Sinusoidal {
            length: length,
            amplitude: amplitude,
            mean: mean,
            step: step,
            phase: (phase - delay as f64 * step) % pi2,
            i: 0,
        }
    }

    /// Constructs a default sinusoidal wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Sinusoidal;
    ///
    /// let x = Sinusoidal::default(10, 8.0, 2.0, 1.0).collect::<Vec<f64>>();
    /// assert_eq!(x,
    ///     [0.0, 1.0, 0.00000000000000012246467991473532,
    ///     -1.0, -0.00000000000000024492935982947064, 1.0,
    ///     0.00000000000000036739403974420594, -1.0,
    ///     -0.0000000000000004898587196589413, 1.0]);
    /// ```
    pub fn default(length: usize,
                   sampling_rate: f64,
                   frequency: f64,
                   amplitude: f64)
                   -> Sinusoidal {

        Self::new(length, sampling_rate, frequency, amplitude, 0.0, 0.0, 0)
    }
}

impl Iterator for Sinusoidal {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.i == self.length {
            None
        } else {
            let x = self.mean + self.amplitude * (self.phase + self.i as f64 * self.step).sin();
            self.i += 1;
            Some(x)
        }
    }
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
#[deprecated(since="0.8.0", note="please use `Sinusoidal::default` instead")]
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
#[deprecated(since="0.8.0", note="please use `Sinusoidal::new` instead")]
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

/// Finite iterator returning floats forming a square wave starting
/// with the high phase
pub struct Square {
    periodic: Periodic,
    high_duration: f64,
    high_value: f64,
    low_value: f64,
}

impl Square {
    /// Constructs a new square wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Square;
    ///
    /// let x = Square::new(12, 3, 7, 1.0, -1.0, 1).collect::<Vec<f64>>();
    /// assert_eq!(x, [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0])
    /// ```
    pub fn new(length: usize,
               high_duration: i64,
               low_duration: i64,
               high_value: f64,
               low_value: f64,
               delay: i64)
               -> Square {

        let duration = (high_duration + low_duration) as f64;
        Square {
            periodic: Periodic::new(length, 1.0, 1.0 / duration, duration, 0.0, delay),
            high_duration: high_duration as f64,
            high_value: high_value,
            low_value: low_value,
        }
    }
}

impl Iterator for Square {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.periodic.next().and_then(|x| {
            if x < self.high_duration {
                Some(self.high_value)
            } else {
                Some(self.low_value)
            }
        })
    }
}

/// Finite iterator returning floats forming a triangle wave
/// starting with the raise phase from the lowest sample
pub struct Triangle {
    periodic: Periodic,
    raise_duration: f64,
    raise: f64,
    fall: f64,
    high_value: f64,
    low_value: f64,
}

impl Triangle {
    /// Constructs a new triangle wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::generate::Triangle;
    ///
    /// # fn main() {
    /// let x = Triangle::new(12, 4, 7, 1.0, -1.0, 1).collect::<Vec<f64>>();
    /// let expected: [f64; 12] = [-0.714, -1.0, -0.5, 0.0, 0.5, 1.0, 0.714, 0.429, 0.143, -0.143, -0.429, -0.714];
    /// for (&left, &right) in x.iter().zip(expected.iter()) {
    ///     assert_almost_eq!(left, right, 1e-3);
    /// }
    /// # }
    /// ```
    pub fn new(length: usize,
               raise_duration: i64,
               fall_duration: i64,
               high_value: f64,
               low_value: f64,
               delay: i64)
               -> Triangle {

        let duration = (raise_duration + fall_duration) as f64;
        let height = high_value - low_value;
        Triangle {
            periodic: Periodic::new(length, 1.0, 1.0 / duration, duration, 0.0, delay),
            raise_duration: raise_duration as f64,
            raise: height / raise_duration as f64,
            fall: height / fall_duration as f64,
            high_value: high_value,
            low_value: low_value,
        }
    }
}

impl Iterator for Triangle {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.periodic.next().and_then(|x| {
            if x < self.raise_duration {
                Some(self.low_value + x * self.raise)
            } else {
                Some(self.high_value - (x - self.raise_duration) * self.fall)
            }
        })
    }
}