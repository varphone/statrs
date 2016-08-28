/// Evaluates a numerically stable
/// series summation where `next` returns the
/// next summand in the series
pub fn series<F>(mut next: F) -> f64
    where F: FnMut() -> f64
{
    let factor = (1 << 16) as f64;
    let mut comp = 0.0;
    let mut sum = next();

    loop {
        let cur = next();
        let y = cur - comp;
        let t = sum + y;
        comp = t - sum - y;
        sum = t;
        if sum.abs() >= (factor * cur).abs() {
            break;
        }
    }
    sum
}