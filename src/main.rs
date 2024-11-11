extern crate rand;
extern crate rayon;

use std::f64;
use rand::Rng;
use rayon::prelude::*;

#[derive(Debug)]
struct Point {
    x: f64,
    y: f64,
}

fn find_equidistant_point(p1: &Point, p2: &Point) -> Option<Point> {
    // Try each border of the unit square
    let borders = [
        (0.0, 'x'), (1.0, 'x'),  // Left and right borders
        (0.0, 'y'), (1.0, 'y'),  // Bottom and top borders
    ];

    for (coord, axis) in borders {
        match axis {
            'x' => {
                let x = coord;
                let a = 1.0;
                let b = -2.0 * (p1.y + p2.y);
                let c = p1.y.powi(2) - p2.y.powi(2) + 
                       (x - p1.x).powi(2) - (x - p2.x).powi(2);
                
                let discriminant = b.powi(2) - 4.0 * a * c;
                if discriminant >= 0.0 {
                    let y1 = (-b + discriminant.sqrt()) / (2.0 * a);
                    let y2 = (-b - discriminant.sqrt()) / (2.0 * a);
                    
                    for y in [y1, y2] {
                        if y >= 0.0 && y <= 1.0 {
                            return Some(Point { x, y });
                        }
                    }
                }
            },
            'y' => {
                let y = coord;
                let a = 1.0;
                let b = -2.0 * (p1.x + p2.x);
                let c = p1.x.powi(2) - p2.x.powi(2) + 
                       (y - p1.y).powi(2) - (y - p2.y).powi(2);
                
                let discriminant = b.powi(2) - 4.0 * a * c;
                if discriminant >= 0.0 {
                    let x1 = (-b + discriminant.sqrt()) / (2.0 * a);
                    let x2 = (-b - discriminant.sqrt()) / (2.0 * a);
                    
                    for x in [x1, x2] {
                        if x >= 0.0 && x <= 1.0 {
                            return Some(Point { x, y });
                        }
                    }
                }
            },
            _ => unreachable!(),
        }
    }
    None
}

fn main() {
    let chunks = 100_000;  // Number of chunks to split the work into
    let trials_per_chunk = 1_000_000;
    let total_trials = chunks as u64 * trials_per_chunk;
    
    let progress_bar = indicatif::ProgressBar::new(chunks as u64);
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
    );

    let positive_results: u64 = (0..chunks)
        .into_par_iter()  // Parallel iterator
        .map(|_| {
            let mut local_positive = 0u64;
            let mut rng = rand::thread_rng();
            
            for _ in 0..trials_per_chunk {
                let p1 = Point {
                    x: rng.gen_range(0.0..=1.0),
                    y: rng.gen_range(0.0..=1.0),
                };
                let p2 = Point {
                    x: rng.gen_range(0.0..=1.0),
                    y: rng.gen_range(0.0..=1.0),
                };

                if find_equidistant_point(&p1, &p2).is_some() {
                    local_positive += 1;
                }
            }
            progress_bar.inc(1);
            local_positive
        })
        .sum();

    progress_bar.finish_with_message("Completed");
    println!(
        "Probability: {:.10} ({} out of {})", 
        positive_results as f64 / total_trials as f64,
        positive_results,
        total_trials
    );
}