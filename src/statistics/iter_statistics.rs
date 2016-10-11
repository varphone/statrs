use std::f64;

pub trait IterStatistics : Iterator<Item=f64> {
    fn abs_min(&mut self) -> f64 {
        let mut min = f64::INFINITY;
        let mut any = false;
        loop {
            match self.next() {
                Some(x) => {
                    let abs = x.abs();
                    if abs < min || abs.is_nan() {
                        min = abs;
                    }
                    any = true;
                }
                None => break
            }
        }
        if any { min } else { f64::NAN }
    }
}