
cimport numpy as np

ctypedef np.int64_t DTYPE_it
ctypedef np.float_t DTYPE_ft

def two_opt_test(np.ndarray[DTYPE_it, ndim=1] route, int i, int k, np.ndarray[DTYPE_ft, ndim=2] distances, int k_max):
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

    """
    cdef float removed = 0
    cdef float added = 0

    if i > 0:
        removed = distances[route[i - 1], route[i]]
        added = distances[route[i - 1], route[k]]

    if k < k_max:
        removed += distances[route[k], route[k + 1]]
        added += distances[route[i], route[k + 1]]

    return added - removed
