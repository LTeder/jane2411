import numpy as np
from typing import Optional
from numba import jit, prange
from tqdm import tqdm

@jit(nopython=True)
def find_equidistant_point(p1: np.ndarray, p2: np.ndarray, side: str) -> Optional[float]:
    """
    Find a point on the specified side that is equidistant from both given points.
    Returns None if no such point exists.
    """
    # Convert to easier variable names for readability
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    
    # Different equations based on which side we're checking
    if side == 'left':  # x = 0
        a = 1.0
        b = -2.0 * (y1 + y2)
        c = (y1**2 - y2**2 + x1**2 - x2**2)
        
    elif side == 'right':  # x = 1
        a = 1.0
        b = -2.0 * (y1 + y2)
        c = (y1**2 - y2**2 + (x1-1.0)**2 - (x2-1.0)**2)
        
    elif side == 'bottom':  # y = 0
        a = 1.0
        b = -2.0 * (x1 + x2)
        c = (x1**2 - x2**2 + y1**2 - y2**2)
        
    else:  # top: y = 1
        a = 1.0
        b = -2.0 * (x1 + x2)
        c = (x1**2 - x2**2 + (y1-1.0)**2 - (y2-1.0)**2)

    # Solve quadratic equation
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return -1.0
        
    sol1 = (-b + np.sqrt(discriminant))/(2*a)
    sol2 = (-b - np.sqrt(discriminant))/(2*a)
    
    # Check if any solution falls within [0,1]
    if 0 <= sol1 <= 1:
        return sol1
    if 0 <= sol2 <= 1:
        return sol2
    
    return -1.0

@jit(nopython=True)
def trial() -> bool:
    # Generate random points
    p1 = np.random.random(2)
    p2 = np.random.random(2)
    
    # Find closest side to p1
    distances = np.array([p1[0], 1.0 - p1[0], p1[1], 1.0 - p1[1]])
    min_idx = np.argmin(distances)
    
    # Convert index to side string
    if min_idx == 0:
        side = 'left'
    elif min_idx == 1:
        side = 'right'
    elif min_idx == 2:
        side = 'bottom'
    else:
        side = 'top'
        
    result = find_equidistant_point(p1, p2, side)
    return result >= 0

@jit(nopython=True, parallel=True)
def run_simulation_batch(batch_size: int) -> int:
    """Run a batch of trials and return the number of positive results."""
    positive_results = 0
    for _ in prange(batch_size):
        if trial():
            positive_results += 1
    return positive_results

def run_simulation_with_progress(total_trials: int, batch_size: int) -> float:
    """Run the simulation with progress tracking using tqdm."""
    positive_results = 0
    batches = total_trials // batch_size
    remainder = total_trials % batch_size
    
    with tqdm(total=total_trials) as pbar:
        for _ in range(batches):
            positive_results += run_simulation_batch(batch_size)
            pbar.update(batch_size)
        
        if remainder > 0:
            positive_results += run_simulation_batch(remainder)
            pbar.update(remainder)
            
    return positive_results / total_trials

if __name__ == "__main__":
    trials = int(1e10)
    batch_size = int(1e7)
    
    rate = run_simulation_with_progress(trials, batch_size)
    print(f"Rate of positive results after {trials} trials: {rate}")