
import numpy as np
from PYME.Analysis.points.traveling_salesperson import two_opt_utils

def calculate_path_length(distances, route):
    """
    Parameters
    ----------
    distances: ndarray
        distance array, for which distances[i, j] is the distance from the ith to the jth point
    route: ndarray
        array of indices defining the path
    """
    return distances[route[:-1], route[1:]].sum()

def two_opt_swap(route, i, k):
    """
    Take everything the same up to i, then reverse i:k, then take k: normally.
    Parameters
    ----------
    route: ndarray
        Path to swap postions in
    i: int
        first swap index
    k: int
        second swap index
    Returns
    -------
    two-opt swapped route

    Notes
    -----
    Returns a copy. Temping to do something in place, e.g. route[i:k + 1] = route[k:i - 1: -1], but the algorithm
    seems to require a copy somewhere anyway, so might as well do it here.

    """
    return np.concatenate([route[:i],  # [start up to first swap position)
                           route[k:i - 1: -1],  # [from first swap to second], reversed
                           route[k + 1:]])  # (everything else]


def two_opt_test(route, i, k, distances, k_max):
    """
    Test to see what distance change we'd get doing a two_opt_swap.
    Take everything the same up to i, then reverse i:k, then take k: normally.


    Parameters
    ----------
    route: ndarray
        Path to swap postions in
    i: int
        first swap index
    k: int
        second swap index
    distances: NxN matrix of float
        distances between points
    k_max: pre-computed maximum value of k  == distances.shape[0] -1

    Returns
    -------
    distance change on swap

    Notes
    -----
    There is a cythonized version in two_opt_utils which is considerably faster.

    """
    removed = 0
    added = 0

    if i > 0:
        removed = distances[route[i - 1], route[i]]
        added = distances[route[i - 1], route[k]]

    if k < k_max:
        removed += distances[route[k], route[k + 1]]
        added += distances[route[i], route[k + 1]]

    return added - removed


def two_opt(distances, epsilon, initial_route=None, fixed_endpoint=False):
    """

    Solves the traveling salesperson problem (TSP) using two-opt swaps to untangle a route.

    Parameters
    ----------
    distances: ndarray
        distance array, which distances[i, j] is the distance from the ith to the jth point
    epsilon: float
        exit tolerence on relative improvement. 0.01 corresponds to 1%
    initial_route: ndarray
        [optional] route to initialize search with. Note that the first position in the route is fixed, but all others
        may vary. If no route is provided, the initial route is the same order the distances array was constructed with.

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
    route = initial_route.astype(int) if initial_route is not None else np.arange(distances.shape[0], dtype=int)
    endpoint_offset = int(fixed_endpoint)

    og_distance = calculate_path_length(distances, route)
    # initialize values we'll be updating
    improvement = 1
    best_distance = og_distance
    k_max = distances.shape[0] - 1
    while improvement > epsilon:
        last_distance = best_distance
        for i in range(1, distances.shape[0] - 2):  # don't swap the first position
            for k in range(i + 1, distances.shape[0] - endpoint_offset):
                d_dist = two_opt_utils.two_opt_test(route, i, k, distances, k_max)
                if d_dist < 0:
                    # do the swap in-place since we tested before we leaped and we don't need the old route
                    route[i:k + 1] = route[k:i - 1: -1]
                    best_distance = best_distance + d_dist

        improvement = (last_distance - best_distance) / last_distance

    return route, best_distance, og_distance


def timeout_two_opt(distances, epsilon, timeout, initial_route=None):
    """

    Solves the traveling salesperson problem (TSP) using two-opt swaps to untangle a route.

    Parameters
    ----------
    distances: ndarray
        distance array, which distances[i, j] is the distance from the ith to the jth point
    epsilon: float
        exit tolerance on relative improvement. 0.01 corresponds to 1%
    timeout: float
        number of seconds to allow computation
    initial_route: ndarray
        [optional] route to initialize search with. Note that the first position in the route is fixed, but all others
        may vary. If no route is provided, the initial route is the same order the distances array was constructed with.

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
    import time
    abort_time = time.time() + timeout
    # start route backwards. Starting point will be fixed, and we want LIFO for fast microscope acquisition
    route = initial_route if initial_route is not None else np.arange(distances.shape[0] - 1, -1, -1)

    og_distance = calculate_path_length(distances, route)
    # initialize values we'll be updating
    improvement = 1
    best_distance = og_distance
    while improvement > epsilon:
        last_distance = best_distance
        for i in range(1, distances.shape[0] - 2):  # don't swap the first position
            if time.time() > abort_time:
                return route, best_distance, og_distance
            for k in range(i + 1, distances.shape[0]):  # allow the last position in the route to vary
                new_route = two_opt_swap(route, i, k)
                new_distance = calculate_path_length(distances, new_route)

                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
        improvement = (last_distance - best_distance) / last_distance

    return route, best_distance, og_distance
