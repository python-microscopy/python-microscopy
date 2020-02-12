
import numpy as np
import multiprocessing
import time
from PYME.util.shmarray import shmarray
from . import two_opt
import logging

logger = logging.getLogger(__name__)

# --------- splitting points into sections ---------

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

# --------- section-linking tools ---------

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
                new_section_order = two_opt.two_opt_swap(section_order, i, k)
                for rev_i in range(i, n_sections):
                    new_reversals = reversal_swap(reversals, rev_i)
                    new_distance = calculate_length_with_reversal(new_section_order, new_reversals, pivot_positions)
                    if new_distance < best_distance:
                        section_order = new_section_order
                        reversals = new_reversals
                        best_distance = new_distance
        improvement = (last_distance - best_distance) / last_distance
    return section_order, reversals, og_distance, best_distance

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

# --------- multiproc target and calling functions ---------

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

        best_route, best_distance, og_distance = two_opt.two_opt(distances, epsilon, route, fixed_endpoint=True)
        # print(best_route)
        master_route[start_route:start_route + counts[ti]] = start_route + best_route
        start_route += counts[ti]
        start_pos += counts[ti]


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
