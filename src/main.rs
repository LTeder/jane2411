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
    // Define borders with their corresponding fixed axis and value
    let borders = [
        ("left", 'x', 0.0, p1.x),
        ("right", 'x', 1.0, 1.0 - p1.x),
        ("bottom", 'y', 0.0, p1.y),
        ("top", 'y', 1.0, 1.0 - p1.y),
    ];

    // Find the closest border to p1
    let (_, axis, fixed_value, _) = borders.iter()
        .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .unwrap();

    compute_equidistant_point(p1, p2, *axis, *fixed_value)
}

fn compute_equidistant_point(p1: &Point, p2: &Point, axis: char, fixed_value: f64) -> Option<Point> {
    if axis == 'x' {
        let x = fixed_value;
        let a = 1.0;
        let b = -2.0 * (p1.y + p2.y);
        let c = p1.y.powi(2) - p2.y.powi(2) + 
                (x - p1.x).powi(2) - (x - p2.x).powi(2);
        
        let discriminant = b.powi(2) - 4.0 * a * c;
        if discriminant >= 0.0 {
            let sqrt_d = discriminant.sqrt();
            let y1 = (-b + sqrt_d) / (2.0 * a);
            let y2 = (-b - sqrt_d) / (2.0 * a);
            
            for y in [y1, y2] {
                if y >= 0.0 && y <= 1.0 {
                    return Some(Point { x, y });
                }
            }
        }
    } else if axis == 'y' {
        let y = fixed_value;
        let a = 1.0;
        let b = -2.0 * (p1.x + p2.x);
        let c = p1.x.powi(2) - p2.x.powi(2) + 
                (y - p1.y).powi(2) - (y - p2.y).powi(2);
        
        let discriminant = b.powi(2) - 4.0 * a * c;
        if discriminant >= 0.0 {
            let sqrt_d = discriminant.sqrt();
            let x1 = (-b + sqrt_d) / (2.0 * a);
            let x2 = (-b - sqrt_d) / (2.0 * a);
            
            for x in [x1, x2] {
                if x >= 0.0 && x <= 1.0 {
                    return Some(Point { x, y });
                }
            }
        }
    }

    None
}

fn main() {
    let chunks = 100_000;  // Number of chunks to split the work into
    let trials_per_chunk = 100_000;
    let total_trials = chunks as u64 * trials_per_chunk;
    
    let progress_bar = indicatif::ProgressBar::new(chunks as u64);
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
    );

    let positive_results: u64 = (0..chunks)
        .into_par_iter()
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
        "Probability: {:.10}", 
        positive_results as f64 / total_trials as f64
    );
}