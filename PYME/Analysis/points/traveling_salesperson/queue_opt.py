
import numpy as np
import time
import threading

class TSPQueue(object):
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
        from PYME.Analysis.points.traveling_salesperson import two_opt, two_opt_utils
        route = initial_route.astype(int)
        endpoint_offset = 0

        # initialize values we'll be updating
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
            print('n sorted: %d' % self.start_ind)
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
        self.route = route
        # return route, best_distance, og_distance

    def __getitem__(self, item):
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
