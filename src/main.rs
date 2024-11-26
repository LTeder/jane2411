extern crate indicatif;
extern crate rand;
extern crate rand_xoshiro;
extern crate rayon;

use std::f64;
use std::sync::atomic::{AtomicU64, Ordering};
use indicatif::ProgressBar;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;


const PI4: f64 = f64::consts::PI / 4.0;

#[inline(always)]
fn do_trial(u: f64, v: f64) -> f64 {
    let (x, y) = if u + v > 1.0 {
        (0.5 * u + v - 0.5, 0.5 * (1.0 - u))
    } else {
        ((0.5 * u) + v, 0.5 * u)
    };
    let x2 = x.powf(2.0);
    let y2 = y.powf(2.0);
    let xi2 = (1.0 - x).powf(2.0);
    PI4 * (2.0 * y2 + x2 + xi2) -
        0.5 * ((y /  x).atan() * (x2  + y2) + (y / (1.0 - x)).atan() * (xi2 + y2))
}


#[inline(always)]
fn main() {
    let chunks: u64 = 5_000_000;
    let trials_per_chunk = 250_000;
    let increment = 10_000;
    let total_trials = chunks * trials_per_chunk;

    let progress_bar = ProgressBar::new(chunks / increment);
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
    );
    let progress = AtomicU64::new(0);

    let results: f64 = (0..chunks)
        .into_par_iter()
        .map(|_| {
            let mut local_positive = 0f64;
            let mut rng = Xoshiro256Plus::seed_from_u64(rand::thread_rng().gen());
            for _ in 0..trials_per_chunk {
                let u: f64 = rng.gen_range(0.0..=1.0);
                let v: f64 = rng.gen_range(0.0..=1.0);
                local_positive += do_trial(u, v)
            }
            let current = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if current % increment == 0 {
                progress_bar.inc(1);
            }
            local_positive
        })
        .sum();

    progress_bar.finish_with_message("Completed");
    println!(
        "Probability after {} trials: {:.12}",
        total_trials,
        results / total_trials as f64
    );
}