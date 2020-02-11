

import numpy as np
import time
from scipy.spatial import distance_matrix

def test_timeout_two_opt():
    from PYME.Analysis.points.traveling_salesperson import two_opt
    n = 2000
    timeout = 5
    x = np.random.rand(n) * 4e3
    y = np.random.rand(n) * 4e3
    positions = np.stack([x, y], axis=1)
    distances = distance_matrix(positions, positions)
    t0 = time.time()
    route, best_distance, og_distance = two_opt.timeout_two_opt(distances, 0.01, timeout)
    elapsed = time.time() - t0
    assert elapsed - timeout < 1.25 * timeout
    assert best_distance < og_distance
    np.testing.assert_array_equal(np.sort(route), np.arange(positions.shape[0], dtype=int))

def test_greedy():
    from PYME.Analysis.points.traveling_salesperson import sort
    n = 500
    x = np.random.rand(n) * 4e3
    y = np.random.rand(n) * 4e3
    positions = np.stack([x, y], axis=1)
    start_ind = np.argmin(positions.sum(axis=1))
    route, og_distance, final_distance = sort.greedy_sort(positions, start_ind)
    uni, counts = np.unique(route, return_counts=True)
    np.testing.assert_array_equal(counts, 1)
    assert final_distance < og_distance

def test_tsp_sort():
    from PYME.Analysis.points.traveling_salesperson import sort
    n = 500
    x = np.random.rand(n) * 4e3
    y = np.random.rand(n) * 4e3
    og_positions = np.stack([x, y], axis=1)
    positions, og_distance, final_distance = sort.tsp_sort(og_positions, return_path_length=True)
    np.testing.assert_array_equal(np.unique(positions), np.unique(og_positions))
    assert final_distance < og_distance
