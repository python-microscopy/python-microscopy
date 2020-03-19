
import numpy as np
cimport numpy as np

ctypedef np.int_t DTYPE_it
ctypedef np.float_t DTYPE_ft

def two_opt_swap(np.ndarray[DTYPE_it, ndim=1] route, int i, int k):
    cdef int ind, ind1
    # cdef int route_length = route.shape[0]

    return np.concatenate([route[:i],  # [start up to first swap position)
                       route[k:i - 1: -1],  # [from first swap to second], reversed
                       route[k + 1:]])  # (everything else]

def two_opt_test(np.ndarray[DTYPE_it, ndim=1] route, int i, int k, np.ndarray[DTYPE_ft, ndim=2] distances, int k_max):
    cdef float removed = 0
    cdef float added = 0

    if i > 0:
        removed = distances[route[i - 1], route[i]]
        added = distances[route[i - 1], route[k]]

    if k < k_max:
        removed += distances[route[k], route[k + 1]]
        added += distances[route[i], route[k + 1]]

    return added - removed

def calculate_path_length(np.ndarray[DTYPE_ft, ndim=2] distances, np.ndarray[DTYPE_it, ndim=1] route):
    """
    Parameters
    ----------
    distances: ndarray
        distance array, for which distances[i, j] is the distance from the ith to the jth point
    route: ndarray
        array of indices defining the path
    """
    cdef float distance = 0
    cdef int ind_r, ind_c, route_length = route.shape[0]

    for ind_r, ind_c in zip(range(route_length - 1), range(1, route_length)):
            distance += distances[route[ind_r], route[ind_c]]

    return distance

def two_opt(np.ndarray[DTYPE_ft, ndim=2] distances, np.ndarray[DTYPE_it, ndim=1] route, float epsilon,
            int endpoint_offset):
    """

    Solves the traveling salesperson problem (TSP) using two-opt swaps to untangle a route.

    Parameters
    ----------
    distances: ndarray
        distance array, which distances[i, j] is the distance from the ith to the jth point
    epsilon: float
        exit tolerence on relative improvement. 0.01 corresponds to 1%
    initial_route: ndarray
        route to initialize search with. Note that the first position in the route is fixed, but all others may vary.
    endpoint_offset: int
        toggles whether endpoint of route is fixed (1) or not (0)

    Returns
    -------
    route: ndarray
        "solved" route
    best_distance: float
        distance of the route
    og_distance: float
        distance of the initial route.

    Notes
    -----
    see https://en.wikipedia.org/wiki/2-opt for pseudo code

    """
    cdef int k_max = distances.shape[0] - 1
    cdef float improvement = 1
    cdef float og_distance = calculate_path_length(distances, route)
    cdef float best_distance = og_distance
    cdef float last_distance
    cdef float d_dist

    while improvement > epsilon:
        last_distance = best_distance
        for i in range(1, distances.shape[0] - 2):  # don't swap the first position
            for k in range(i + 1, distances.shape[0] - endpoint_offset):
                d_dist = two_opt_test(route, i, k, distances, k_max)
                if d_dist < 0:
                    route = two_opt_swap(route, i, k)
                    best_distance = best_distance + d_dist #calculate_path_length(distances, route)

        improvement = (last_distance - best_distance) / last_distance

    return route, best_distance, og_distance