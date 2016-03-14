/// Enumeration of possible errors thrown
/// within the statrs library
pub enum StatsError {
    BadParams,
    ArgMustBePositive(&'static str),
    ArgNotNegative(&'static str),
    ArgIntervalIncl(&'static str),
}
