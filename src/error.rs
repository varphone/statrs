/// Enumeration of possible errors thrown
/// within the statrs library
#[derive(Debug)]
pub enum StatsError {
    BadParams,
    ArgMustBePositive(&'static str),
    ArgNotNegative(&'static str),
    ArgIntervalIncl(&'static str, f64, f64),
}
