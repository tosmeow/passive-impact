use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Exp, Distribution};

pub fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

#[inline]
pub fn sample_exponential<R: Rng>(rng: &mut R, lambda: f64) -> f64 {
    let exp = Exp::new(lambda).unwrap();
    exp.sample(rng)
}

#[inline]
pub fn sample_uniform<R: Rng>(rng: &mut R) -> f64 {
    rng.gen()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeded_rng_reproducible() {
        let mut rng1 = create_rng(Some(42));
        let mut rng2 = create_rng(Some(42));

        for _ in 0..100 {
            assert_eq!(sample_uniform(&mut rng1), sample_uniform(&mut rng2));
        }
    }

    #[test]
    fn test_exponential_positive() {
        let mut rng = create_rng(Some(123));
        for _ in 0..100 {
            let sample = sample_exponential(&mut rng, 1.0);
            assert!(sample > 0.0);
        }
    }
}
