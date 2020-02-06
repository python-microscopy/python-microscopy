
from PYME.localization import traveling_salesperson as tsp
import numpy as np
import time
from scipy.spatial import distance_matrix

def test_timeout_two_opt():
    n = 2000
    timeout = 15
    x = np.random.rand(n) * 4e3
    y = np.random.rand(n) * 4e3
    positions = np.stack([x, y], axis=1)
    distances = distance_matrix(positions, positions)
    t0 = time.time()
    route, best_distance, og_distance = tsp.timeout_two_opt(distances, 0.01, timeout)
    elapsed = time.time() - t0
    assert elapsed - timeout < 1.25 * timeout
    assert best_distance < og_distance
    np.testing.assert_array_equal(np.sort(route), np.arange(positions.shape[0], dtype=int))
