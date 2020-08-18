

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

def test_swap():
    from PYME.Analysis.points.traveling_salesperson.two_opt import two_opt_swap
    route = np.arange(100, dtype=int)
    i = 30
    k = 60
    tos_route = two_opt_swap(route, i, k)
    route[i:k + 1] = route[k:i - 1: -1]
    assert np.array_equal(tos_route, route)

def test_tsp_queue():
    from PYME.Analysis.points.traveling_salesperson.queue_opt import TSPQueue
    from PYME.Analysis.points.traveling_salesperson.two_opt import calculate_path_length

    n = 2000
    x = np.random.rand(n) * 100
    y = np.random.rand(n) * 100
    positions = np.stack([x, y], axis=1)
    sorter = TSPQueue(positions, start=0, epsilon=0.01, t_step=1)
    sorted_positions = []
    for ind in range(n):
        sorted_positions.append((sorter[ind, 0], sorter[ind, 1]))
    sorted_positions = np.asarray(sorted_positions)
    assert (np.unique(sorted_positions) == np.unique(positions)).all()
    dummy_route = np.arange(len(positions), dtype=int)
    unsorted_distance = calculate_path_length(distance_matrix(positions, positions), dummy_route)
    sorted_distance = calculate_path_length(distance_matrix(sorted_positions, sorted_positions), dummy_route)
    assert sorted_distance < unsorted_distance / 10
