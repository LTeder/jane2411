extern crate indicatif;
extern crate rand;
extern crate rand_xoshiro;
extern crate rayon;

use indicatif::ProgressBar;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use std::f64;
use std::sync::atomic::{AtomicU64, Ordering};


struct Point {
    x: f64,
    y: f64,
}

#[inline(always)]
fn find_equidistant_point(p1: &Point, p2: &Point) -> bool {
    let borders = [
        ('x', 0.0, p1.x),
        ('x', 1.0, 1.0 - p1.x),
        ('y', 0.0, p1.y),
        ('y', 1.0, 1.0 - p1.y),
    ];

    let (axis, fixed_value, _) = borders.iter()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    match *axis {
        'x' => compute_equidistant_point_x(p1, p2, *fixed_value),
        'y' => compute_equidistant_point_y(p1, p2, *fixed_value),
        _ => false,
    }
}

#[inline(always)]
fn compute_equidistant_point_x(p1: &Point, p2: &Point, fixed_x: f64) -> bool {
    let mid = Point {
        x: (p1.x + p2.x) / 2.0,
        y: (p1.y + p2.y) / 2.0,
    };
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    if dy.abs() < f64::EPSILON { // p1 and p2 are the same point
        return false;
    }
    let perpendicular_slope = -dx / dy;
    // Calculate y using y = m(x - mid.x) + mid.y
    let y = perpendicular_slope * (fixed_x - mid.x) + mid.y;
    y >= 0.0 && y <= 1.0
}

#[inline(always)]
fn compute_equidistant_point_y(p1: &Point, p2: &Point, fixed_y: f64) -> bool {
    let mid = Point {
        x: (p1.x + p2.x) / 2.0,
        y: (p1.y + p2.y) / 2.0,
    };
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    if dx.abs() < f64::EPSILON { // p1 and p2 have the same x-coordinate
        return false;
    }
    let perpendicular_slope = -dx / dy;
    let x = (fixed_y - mid.y) / perpendicular_slope + mid.x;
    x >= 0.0 && x <= 1.0
}

fn main() {
    let chunks = 10_000_000;
    let trials_per_chunk = 250_000;
    let total_trials = chunks as u64 * trials_per_chunk;

    let progress_bar = ProgressBar::new(chunks as u64);
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
    );

    let progress = AtomicU64::new(0);

    let positive_results: u64 = (0..chunks)
        .into_par_iter()
        .map(|_| {
            let mut local_positive = 0u64;
            let mut rng = Xoshiro256Plus::seed_from_u64(rand::thread_rng().gen());

            for _ in 0..trials_per_chunk {
                let p1 = Point {
                    x: rng.gen_range(0.0..=1.0),
                    y: rng.gen_range(0.0..=1.0),
                };
                let p2 = Point {
                    x: rng.gen_range(0.0..=1.0),
                    y: rng.gen_range(0.0..=1.0),
                };

                if find_equidistant_point(&p1, &p2) {
                    local_positive += 1;
                }
            }

            let current = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if current % 1000 == 0 {
                progress_bar.inc(1000);
            }

            local_positive
        })
        .sum();

    // Finalize any remaining progress
    let remaining = chunks as u64 - progress.load(Ordering::Relaxed);
    if remaining > 0 {
        progress_bar.inc(remaining);
    }

    progress_bar.finish_with_message("Completed");
    println!(
        "Probability after {} trials: {:.10}",
        total_trials,
        positive_results as f64 / total_trials as f64
    );
}