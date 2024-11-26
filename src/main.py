import numpy as np
from numba import njit, prange
from tqdm import trange

@njit
def find_equidistant_point(p1, p2):
    """
    Determine if an equidistant point exists on the closest side to p1 within [0,1].
    """
    x_min = p1[0]
    x_max = 1.0 - p1[0]
    y_min = p1[1]
    y_max = 1.0 - p1[1]

    # Identify the closest border
    min_val = x_min
    axis = 'x'
    fixed_value = 0.0

    if x_max < min_val:
        min_val = x_max
        axis = 'x'
        fixed_value = 1.0

    if y_min < min_val:
        min_val = y_min
        axis = 'y'
        fixed_value = 0.0

    if y_max < min_val:
        axis = 'y'
        fixed_value = 1.0

    if axis == 'x':
        return compute_equidistant_point_x(p1, p2, fixed_value)
    elif axis == 'y':
        return compute_equidistant_point_y(p1, p2, fixed_value)
    else:
        return False

@njit
def compute_equidistant_point_x(p1, p2, fixed_x):
    mid_x = 0.5 * (p1[0] + p2[0])
    mid_y = 0.5 * (p1[1] + p2[1])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dy == 0.0:
        return False

    perpendicular_slope = -dx / dy
    y = perpendicular_slope * (fixed_x - mid_x) + mid_y
    return 0.0 <= y <= 1.0

@njit
def compute_equidistant_point_y(p1, p2, fixed_y):
    mid_x = 0.5 * (p1[0] + p2[0])
    mid_y = 0.5 * (p1[1] + p2[1])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0.0:
        return False

    perpendicular_slope = -dx / dy
    if perpendicular_slope == 0.0:
        return False

    x = (fixed_y - mid_y) / perpendicular_slope + mid_x
    return 0.0 <= x <= 1.0

@njit
def trial():
    p1 = np.random.random(2)
    p2 = np.random.random(2)
    return find_equidistant_point(p1, p2)

@njit(parallel=True)
def run_simulation_batch(batch_size):
    positive_results = 0
    for _ in prange(batch_size):
        if trial():
            positive_results += 1
    return positive_results

def run_simulation_with_progress(batches, batch_size):
    positive_results = 0
    remainder = batches % batch_size

    for _ in trange(batches):
        positive_results += run_simulation_batch(batch_size)
    if remainder > 0:
        positive_results += run_simulation_batch(remainder)
    return positive_results / batches / batch_size

if __name__ == "__main__":
    trials = 100_000
    batch_size = 100_000
    rate = run_simulation_with_progress(trials, batch_size)
    print(f"Probability after {trials*batch_size} trials: {rate:.10f}")