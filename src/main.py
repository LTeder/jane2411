import numpy as np
from numba import njit, prange
from tqdm import trange


PI4 = np.pi / 4.0
@njit
def do_trial(u, v):
    if u + v > 1.0:
        x = 0.5 * u + v - 0.5
        y = 0.5 * (1.0 - u)
    else:
        x = 0.5 * u + v
        y = 0.5 * u
    x2 = x ** 2
    y2 = y ** 2
    xi2 = (1.0 - x) ** 2
    return PI4 * (2.0 * y2 + x2 + xi2) - 0.5 * (
        (np.arctan(y / x) * (x2 + y2)) + (np.arctan(y / (1.0 - x)) * (xi2 + y2)))

@njit(parallel=True)
def run_simulation(chunks, trials_per_chunk):
    total = 0.0
    for i in prange(chunks):
        local_positive = 0.0
        for _ in range(trials_per_chunk):
            u = np.random.random()
            v = np.random.random()
            local_positive += do_trial(u, v)
        total += local_positive
    return total

if __name__ == "__main__":
    batches = 1_000_000
    bs = 100_000
    increment = 10_000
    total_trials = batches * bs
    progress_steps = batches // increment
    results = 0.0
    for _ in trange(progress_steps):
        results += run_simulation(increment, bs)
    rate = results / total_trials
    print(f"Probability after {batches * bs} trials: {rate:.12f}")