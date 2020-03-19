
import numpy as np
import time
import threading

class TSPQueue(object):
    """
    Object which masquerades as an n x 2 positions array for use in position queuing. Delays are added in the
    __getitem__ call according to the t_step attribute which allows positions to be sorted in the background while
    positions near the beginning of the route are already accessible.
    """
    def __init__(self, positions, start=0, epsilon=0.01, t_step=1):
        from scipy.spatial import distance_matrix, cKDTree
        from PYME.Analysis.points.traveling_salesperson import sort

        self.epsilon = epsilon
        self.t_step = t_step

        self.distances = distance_matrix(positions, positions)

        if np.isscalar(start):
            start_index = int(start)
        else:
            kdt = cKDTree(positions)
            d, start_index = kdt.query(start)

        # bootstrap with a greedy sort
        self.initial_route, self.initial_path_length, self.greedy_sort_dist = sort.greedy_sort(positions, start_index,
                                                                                               self.distances)
        self.route = [self.initial_route[0]]
        self.t0 = time.time()
        self.sort_thread = threading.Thread(target=self.queue_two_opt, args=(self.initial_route,
                                                                             self.epsilon, self.t_step, self.distances,
                                                                             self.greedy_sort_dist))
        self.sort_thread.start()
        self.positions = positions
        self.shape = positions.shape

    @property
    def start_ind(self):
        return len(self.route)

    def queue_two_opt(self, initial_route, epsilon, t_step, distances, og_distance):
        """
        Solves the traveling salesperson problem (TSP) using two-opt swaps to untangle a route. Does so in a way which
        is convenient for quickly accessing the beginning of the sorted list, as t_step is used to periodically advance
        the first index which can be moved by the sort

        Parameters
        ----------
        initial_route: ndarray
            [optional] route to initialize search with. Note that the first position in the route is fixed, but all others
            may vary. If no route is provided, the initial route is the same order the distances array was constructed with.
        epsilon: float
            exit tolerence on relative improvement. 0.01 corresponds to 1%
        t_step: float
            average time after which to advance an index. A single step can easily be longer if the nested for loop
            takes a considerable amount of time, after which multiple indices will be moved to the 'sorted' route to
            catch up.
        distances: ndarray
            distance array, which distances[i, j] is the distance from the ith to the jth point
        og_distance: float
            path-length of initial_route

        """
        from PYME.Analysis.points.traveling_salesperson import two_opt_utils
        route = initial_route.astype(int)
        endpoint_offset = 0
        improvement = 1
        best_distance = og_distance
        k_max = distances.shape[0] - 1
        t0 = self.t0
        while improvement > epsilon:
            last_distance = best_distance
            t1 = time.time()
            steps = np.ceil((t1 - t0)/t_step)
            t0 = t1  # prepare for next go-round
            if steps:
                for ind in range(int(steps)):
                    # append to master route
                    self.route.append(route[self.start_ind])
            print('n sorted: %d, after %f s' % (self.start_ind, (time.time() - self.t0)))
            for i in range(self.start_ind, distances.shape[0] - 2):
                # start swapping at the current 'start' of our route
                for k in range(i + 1, distances.shape[0] - endpoint_offset):
                    d_dist = two_opt_utils.two_opt_test(route, i, k, distances, k_max)
                    if d_dist < 0:
                        # do the swap in-place since we tested before we leaped and we don't need the old route
                        route[i:k + 1] = route[k:i - 1: -1]
                        best_distance = best_distance + d_dist

            improvement = (last_distance - best_distance) / last_distance
        # add the rest of the route
        self.route.extend(route[self.start_ind:])

    def __getitem__(self, item):
        """
        Get a position(s), sorted in an effort to minimize path length. Will not return immediately - the two-opt
        sorting is allowed to continue until the next time it wants to move the first index at which point it will check
        to see if it has delayed too long and needs to simply add a couple positions from the greedy-sort bootstrapped
        route to the final route, which can be accessed from this call.
        """
        # our positions array is shape n, 2
        if len(item) == 1:
            n, xy = item, slice(None, None, None)
        elif len(item) == 2:
            n, xy = item
        else:
            raise IndexError

        if n > len(self.positions) or n < 0:  # don't allow wrapping for now
            raise IndexError

        while n > self.start_ind - 1:
            time.sleep(self.t_step)

        return self.positions[self.route, :][item]
