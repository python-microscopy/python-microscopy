
import numpy as np
import logging
from . import two_opt

logger = logging.getLogger(__name__)


def greedy_sort(positions, start_index=0, distances=None):
    """
    Nearest-neighbors path optimization.

    Parameters
    ----------
    positions: 2darray
        Positions array, size n x 2
    start_index: int
        Index to start path from.
    distances: ndarray
        [optional] distance array, for which distances[i, j] is the distance from the ith to the jth point. Will be
        calculated if not provided.

    Returns
    -------
    route: 1darray
        array of indices defining the path
    og_distance: float
        distance of path traversing positions in their original order
    final_distance: float
        final distance after path optimization
    """
    if distances is None:
        from scipy.spatial import distance_matrix
        distances = distance_matrix(positions, positions)
    route = np.arange(distances.shape[0], dtype=int)
    og_distance = two_opt.calculate_path_length(distances, route)
    visited = np.zeros(distances.shape[0], dtype=bool)
    maxf = np.finfo(float).max
    for ind in range(distances.shape[0]):
        visited[start_index] = True
        next_index = np.argmin(distances[start_index, :] + visited * maxf)
        # next_index += np.sum(~unvisited[:next_index])
        route[ind] = start_index
        start_index = next_index

    final_distance = two_opt.calculate_path_length(distances, route)
    return route, og_distance, final_distance

def tsp_sort(positions, start=0, epsilon=0.01, return_path_length=False):
    """
    Parameters
    ----------
    positions: 2darray
        Positions array, size n x 2
    start_index: int or two-tuple
        starting index as int or two-tuple of x, y position to start the path from.

    Returns
    -------
    sorted_positions: 2darray
        Positions array sorted by the optimized route, size n x 2
    """
    from scipy.spatial import distance_matrix, cKDTree

    distances = distance_matrix(positions, positions)

    if np.isscalar(start):
        start_index = int(start)
    else:
        kdt = cKDTree(positions)
        d, start_index = kdt.query(start)

    # bootstrap with a greedy sort
    route, ogd, gsd = greedy_sort(positions, start_index, distances)
    route, final_distance, gsd = two_opt.two_opt(distances, epsilon, route)
    if return_path_length:
        return positions[route, :], ogd, final_distance
    return positions[route, :]
