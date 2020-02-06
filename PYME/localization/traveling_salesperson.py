
import multiprocessing
from PYME.util.shmarray import shmarray
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
    og_distance = calculate_path_length(distances, route)
    visited = np.zeros(distances.shape[0], dtype=bool)
    maxf = np.finfo(float).max
    for ind in range(distances.shape[0]):
        visited[start_index] = True
        next_index = np.argmin(distances[start_index, :] + visited * maxf)
        # next_index += np.sum(~unvisited[:next_index])
        route[ind] = start_index
        start_index = next_index

    final_distance = calculate_path_length(distances, route)
    return route, og_distance, final_distance


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
    fixed_endpoint: bool
        Flag to allow endpoint to be swapped or remain as defined in `initial_route` or default route.

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
    # start route backwards. Starting point will be fixed, and we want LIFO for fast microscope acquisition
    route = initial_route if initial_route is not None else np.arange(distances.shape[0] - 1, -1, -1)

    endpoint_offset = int(fixed_endpoint)

    og_distance = calculate_path_length(distances, route)
    # initialize values we'll be updating
    improvement = 1
    best_distance = og_distance
    while improvement > epsilon:
        last_distance = best_distance
        for i in range(1, distances.shape[0] - 2):  # don't swap the first position
            for k in range(i + 1, distances.shape[0] - endpoint_offset):  # allow the last position in the route to vary
                new_route = two_opt_swap(route, i, k)
                new_distance = calculate_path_length(distances, new_route)

                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
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

def reversal_swap(reversals, ind):
    """
    Flag a section as being reversed, returning a copy.
    Parameters
    ----------
    reversals: ndarray
        array of boolean flags denoting whether a segment is reversed (True) or not
    ind: int
        index to swap

    Returns
    -------
    new_reversals: ndarray
        copy of input reversals but with ind "not"-ed

    """
    new_reversals = np.copy(reversals)
    new_reversals[ind] = ~new_reversals[ind]
    return new_reversals

def calc_dist(p0, p1):
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def calculate_length_with_reversal(order, reversals, positions):
    """
    Parameters
    ----------
    order: ndarray
        section order
    reversals: ndarray
        array of bool, forward (False) and reverse (True)
    positions: ndarray
        pivot positions, sorted such that [::2] yields the positions of one end of each section.
    Returns
    -------
    length: float
        cumulative length _between_ sections
    """
    porder = np.repeat(2 * order, 2) + np.concatenate([[1, 0] if rev else [0, 1] for rev in reversals])
    ordered_positions = positions[porder]
    return np.sqrt(((ordered_positions[1:-2:2] - ordered_positions[2::2]) ** 2).sum(axis=1)).sum()

def reversal_two_opt(section_ids, pivot_positions, epsilon):
    """
    Parameters
    ----------
    pivot_indices: ndarray
        sorted indices (into original position array) of start and end points for each section
    section_ids: ndarray
        same shape as pivot indices, encoding which section each pivot is in
    epsilon: float
        relative improvement exit criteria
    Returns
    -------
    """
    sections = np.unique(section_ids)
    n_sections = len(sections)
    section_order = sections
    reversals = np.zeros_like(sections, dtype=bool)
    improvement = 1
    best_distance = calculate_length_with_reversal(section_order, reversals, pivot_positions)
    og_distance = best_distance
    while improvement > epsilon:
        last_distance = best_distance
        for i in range(1, n_sections - 1):  # already know section 0 is first, and it is forward
            for k in range(i + 1, n_sections):
                new_section_order = two_opt_swap(section_order, i, k)
                for rev_i in range(i, n_sections):
                    new_reversals = reversal_swap(reversals, rev_i)
                    new_distance = calculate_length_with_reversal(new_section_order, new_reversals, pivot_positions)
                    if new_distance < best_distance:
                        section_order = new_section_order
                        reversals = new_reversals
                        best_distance = new_distance
        improvement = (last_distance - best_distance) / last_distance
    return section_order, reversals, og_distance, best_distance

def two_opt_section(positions, start_section, counts, n_tasks, epsilon, master_route):
    """
    Perform two-opt TSP on (potentially) multiple sections.

    Parameters
    ----------
    positions: ndarray
        positions, shape (n_points, 2), where n_points are just the positions for the tasks this call is responsible for
    start_section: int
        index denoting where the first position in the input positions belongs in the master_route array
    counts: ndarray
        array of counts corresponding to the number of positions in each sorting task
    n_tasks: int
        number of sorting tasks to execute in this function call. Each task will be sorted independently and shoved into
        the master_route array in the order the tasks are executed.
    epsilon: float
        relative improvement exit criteria for sorting
    master_route: shmarray
        output array for the sorted tasks

    """
    from scipy.spatial import distance_matrix
    start_pos = 0
    start_route = start_section
    for ti in range(n_tasks):
        pos = positions[start_pos: start_pos + counts[ti]]
        distances = distance_matrix(pos, pos)

        # start on a corner, rather than center
        route = np.argsort(pos[:, 0] + pos[:, 1])
        # print('route %s' % (route,))

        best_route, best_distance, og_distance = two_opt(distances, epsilon, route, fixed_endpoint=True)
        # print(best_route)
        master_route[start_route:start_route + counts[ti]] = start_route + best_route
        start_route += counts[ti]
        start_pos += counts[ti]

def split_points_kmeans(positions, points_per_chunk):
    """
    Parameters
    ----------
    positions: ndarray
        Positions array, size n x 2
    points_per_chunk: Int
        Number of points desired to be in each chunk that a two-opt algorithm is run on. Larger chunks tend toward more
        ideal paths, but much larger computational complexity.

    Returns
    -------
    section: ndarray
        array of int denoting section assignment
    n_sections: int
        number of sections
    """
    from sklearn.cluster import KMeans
    k = int(np.ceil(positions.shape[0] / points_per_chunk))
    section = KMeans(k).fit_predict(positions)
    # put start section on lower left corner
    llc_section = section[np.argmin(positions.sum(axis=1))]
    llc_mask = section == llc_section
    section[section == 0] = llc_section
    section[llc_mask] = 0
    return section, k


def split_points_by_grid(positions, points_per_chunk):
    """
    Assuming uniform density, separate points using a grid

    Parameters
    ----------
    positions: ndarray
        Positions array, size n x 2
    points_per_chunk: Int
        Number of points desired to be in each chunk that a two-opt algorithm is run on. Larger chunks tend toward more
        ideal paths, but much larger computational complexity.

    Returns
    -------
    section: ndarray
        array of int denoting grid assignment
    n_sections: int
        number of sections in the grid
    """
    # assume density is uniform
    x_min, y_min = positions.min(axis=0)
    x_max, y_max = positions.max(axis=0)

    sections_per_side = int(np.sqrt((positions.shape[0] / points_per_chunk)))
    size_x = (x_max - x_min) / sections_per_side
    size_y = (y_max - y_min) / sections_per_side

    # bin points into our "pixels"
    X = np.round(positions[:, 0] / size_x).astype(int)
    Y = np.round(positions[:, 1] / size_y).astype(int)

    # number the sections
    section = X + Y * (Y.max() + 1)
    # keep all section numbers positive, starting at zero
    section -= section.min()
    n_sections = int(section.max() + 1)
    return section, n_sections

def tsp_chunk_two_opt_multiproc(positions, epsilon, points_per_chunk, n_proc=1):
    # divide points spatially
    positions = positions.astype(np.float32)
    section, n_sections = split_points_kmeans(positions, points_per_chunk)
    I = np.argsort(section)
    section = section[I]
    positions = positions[I, :]

    # split out points
    n_cpu = n_proc if n_proc > 0 else multiprocessing.cpu_count()
    tasks = int(n_sections / n_cpu) * np.ones(n_cpu, 'i')
    tasks[:int(n_sections % n_cpu)] += 1

    if positions.shape[0] < np.iinfo(np.uint16).max:
        route = shmarray.zeros(positions.shape[0], dtype=np.uint16)
    else:
        route = shmarray.zeros(positions.shape[0], dtype='i')

    uni, counts = np.unique(section, return_counts=True)
    logger.debug('%d points total, section counts: %s' % (counts.sum(), (counts,)))
    if (counts > 1000).any():
        logger.warning('%d counts in a bin, traveling salesperson algorithm may be very slow' % counts.max())

    ind_task_start = 0
    ind_pos_start = 0
    processes = []

    cumcount = counts.cumsum()
    cumtasks = tasks.cumsum()
    print('%d tasks' % cumtasks[-1])
    t = time.time()
    if n_cpu == 1:
        two_opt_section(positions, 0, counts, tasks[0], epsilon, route)
    else:
        for ci in range(n_cpu):
            ind_task_end = cumtasks[ci]
            ind_pos_end = cumcount[ind_task_end - 1]

            subcounts = counts[ind_task_start: ind_task_end]

            p = multiprocessing.Process(target=two_opt_section,
                                        args=(positions[ind_pos_start:ind_pos_end, :],
                                              ind_pos_start,
                                              subcounts,
                                              tasks[ci], epsilon, route))
            p.start()
            processes.append(p)
            ind_task_start = ind_task_end
            ind_pos_start = ind_pos_end

        [p.join() for p in processes]
    print('Chunked TSPs finished after ~%.2f s, connecting chunks' % (time.time() - t))

    sorted_pos = positions[route, :]
    # make cuts at the corner of each section
    new_sections = np.empty_like(section)
    n_new_sections = 0
    start = 0
    cut_positions = []
    for sind in range(n_sections):  # we got section 0 for free with the copy
        section_count = counts[sind]
        pos = sorted_pos[start: start + section_count]
        corner0, corner1 = np.argsort(pos[:, 0] + pos[:, 1])[np.array([0, -1])] + start
        corner2, corner3 = np.argsort(pos[:, 0] - pos[:, 1])[np.array([0, -1])] + start
        corners = np.sort([corner0, corner1, corner2, corner3])
        cut_positions.extend([corners[0], corners[1] - 1, corners[1], corners[2] - 1, corners[2], corners[3] - 1])
        for ci in range(3):
            new_sections[corners[ci]:corners[ci + 1] + 1] = n_new_sections
            n_new_sections += 1

        start += section_count
        # label_start +=
    cut_positions[-1] += 1  # move the last corner end of whole thing
    cut_positions = np.sort(cut_positions)
    # np.testing.assert_array_equal(new_sections[cut_positions], np.repeat(np.arange(n_new_sections), 2))

    t = time.time()
    print('linking sections')
    linked_route = link_route(sorted_pos, cut_positions, new_sections, epsilon)
    print('sections linked in %.2f s' % (time.time() - t))

    # np.testing.assert_array_equal(sorted_pos[linked_route], positions[route][linked_route])
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # colors = cm.get_cmap('prism', n_new_sections)
    # plt.figure()
    # plt.plot(sorted_pos[linked_route, 0], sorted_pos[linked_route, 1], color='k')
    # for pi in range(len(section)):
    #     plt.scatter(sorted_pos[pi, 0], sorted_pos[pi, 1], marker='$' + str(new_sections[pi]) + '$',
    #                 color=colors(new_sections[pi]))
    # plt.show()

    # don't forget the linked route sorts the first-pass route, which sorts the section-sorted positions -> take care of that here
    return I[route][linked_route]

def link_route(positions, cut_indices, sections, epsilon):
    """
    Optimize section order and direction with a two-opt + reversal swap algorithm

    Parameters
    ----------
    positions: 2darray
        Positions array, size n x 2
    cut_indices: 1darray
        first and last index for each section
    sections: 1darray
        denotes section assignment for each point in `positions`
    epsilon: float
        relative improvement exit criteria for two-opt
    """
    route = np.arange(len(positions), dtype=int)
    uni, counts = np.unique(sections, return_counts=True)
    n_sections = len(uni)
    cumcount = counts.cumsum()
    section_order, reversals, og_distance, best_distance = reversal_two_opt(sections[cut_indices],
                                                                            positions[cut_indices], epsilon / 1e3)
    print('unoptimized linking distance: %.0f, optimized: %.0f' % (og_distance, best_distance))

    final_route = np.copy(route)
    start = cumcount[0]
    # new_link_indices = []  # uncomment for plotting
    for sind in range(1, n_sections):  # we got section 0 for free with the copy
        cur_section = section_order[sind]
        section_count = counts[cur_section]
        if reversals[sind]:
            final_route[start: start + section_count] = route[cumcount[cur_section - 1]:cumcount[cur_section]][::-1]
        else:
            final_route[start: start + section_count] = route[cumcount[cur_section - 1]:cumcount[cur_section]]
        # new_link_indices.append(start)  # uncomment for plotting
        # new_link_indices.append(start + section_count - 1)  # uncomment for plotting
        start += section_count

    return final_route  #, new_link_indices