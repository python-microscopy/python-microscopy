import numpy as np
from copy import deepcopy

import marching_cubes


class DualMarchingCubes(marching_cubes.MarchingCubes):
    def __init__(self, isolevel=0):
        marching_cubes.MarchingCubes.__init__(self, isolevel)
        self._ot = None

    def set_octree(self, ot):
        self._ot = ot  # Assign the octree

        # Make vertices/values a list instead of None
        self.vertices = []
        self.values = []

        self.node_proc(self._ot._nodes[0])  # Create the dual grid

        # Make vertices/values np arrays
        self.vertices = np.vstack(self.vertices).astype('float64')
        self.values = np.vstack(self.values).astype('float64')

    def position_empty_node(self, n0, n1, shift):
        """
        Function that considers nodes pairwise to replace 0-node if one of the nodes is the 0-node and the self._other is nself._ot.

        Parameters
        ----------
        n0 : _octree._node
            A node in the same subdivision as n1
        n1 : _octree._node
            A node in the same subdivision as n0
        shift : list
            List of x, y, z representing unit vector to move from n0 to n1 in 3-space
        """
        # We need to replace any instances of the 0-node with an empty node
        # corresponding to a position directly across from the self._other
        # node. Essentially, we rebuild this portion of the octree.
        n0_root_mask = n0['depth'] == 0
        n1_root_mask = n1['depth'] == 0
        if np.sum(n0_root_mask) > 0:
            inds = np.where(n0_root_mask)
            empty_node = np.zeros_like(n0[inds])
            empty_node['nPoints'] = self._ot._nodes[n1[inds]['parent']]['nPoints']/8.
            empty_node['depth'] = n1[inds]['depth']
            empty_node['centre'] = n1[inds]['centre'] + np.vstack(
                self._ot.box_size(n1[inds]['depth'])).T * np.array(shift) * -1
            n0[inds] = empty_node
        if np.sum(n1_root_mask) > 0:
            inds = np.where(n1_root_mask)
            empty_node = np.zeros_like(n1[inds])
            empty_node['nPoints'] = self._ot._nodes[n0[inds]['parent']]['nPoints'] / 8.
            empty_node['depth'] = n0[inds]['depth']
            empty_node['centre'] = n0[inds]['centre'] + np.vstack(
                self._ot.box_size(n0[inds]['depth'])).T * np.array(shift)
            n1[inds] = empty_node
        return n0, n1

    def node_proc(self, nodes):
        """
        Apply operations per octree node.

        Parameters
        ----------
        nodes : _octree._node
            Octree node(s)

        Returns
        -------
        None
        """
        # If we have a single element passed to nodes, covert it to an array
        if not nodes.shape:
            nodes = np.array([nodes])

        # If the node is nself._ot subdivided, remove it from our array
        # DB: can this be moved outside the recursive code - e.g. to set_octree?
        # DB: Actually, as node_proc doesn't use the return of node_proc(children) can we replace the recursion here completely with a loop
        # DB: over our flat nodes?
        # DB: This raises another issue - If this is equvalent to a flat loop over nodes (and we don't use the vertices from deeper
        # DB: in the tree), how do we actually generate the edges which span between nodes?
        nodes = nodes[np.sum(nodes['children'], axis=1) > 0]

        # Perform operations on all subdivided nodes
        if nodes.size > 0:
            # Grab all available children and apply node_proc
            # Can't apply node_proc to the 0-node again or we'll get infinite recursion
            children = self._ot._nodes[nodes['children'][nodes['children'] > 0]]
            self.node_proc(children)

            # We nself._ot call face_proc_<plane> and edge_proc_<plane> on nodes in a specific order
            # so we can track where subdivided nodes will be in reference to adjacent subdivided
            # nodes
            
            # DB: do we need to worry about sparse children for all of these calls? 
            # At present you are any non-occupied nodes will be interpreted as the root node

            # Call self.face_proc_xy on the correct nodes
            self.face_proc_xy(self._ot._nodes[nodes['children'][:, 0]],
                         self._ot._nodes[nodes['children'][:, 4]])
            self.face_proc_xy(self._ot._nodes[nodes['children'][:, 2]],
                         self._ot._nodes[nodes['children'][:, 6]])
            self.face_proc_xy(self._ot._nodes[nodes['children'][:, 1]],
                         self._ot._nodes[nodes['children'][:, 5]])
            self.face_proc_xy(self._ot._nodes[nodes['children'][:, 3]],
                         self._ot._nodes[nodes['children'][:, 7]])

            # Call self.face_proc_xz on the correct nodes
            self.face_proc_xz(self._ot._nodes[nodes['children'][:, 0]],
                         self._ot._nodes[nodes['children'][:, 2]])
            self.face_proc_xz(self._ot._nodes[nodes['children'][:, 1]],
                         self._ot._nodes[nodes['children'][:, 3]])
            self.face_proc_xz(self._ot._nodes[nodes['children'][:, 4]],
                         self._ot._nodes[nodes['children'][:, 6]])
            self.face_proc_xz(self._ot._nodes[nodes['children'][:, 5]],
                         self._ot._nodes[nodes['children'][:, 7]])

            # Call self.face_proc_yz on the correct nodes
            self.face_proc_yz(self._ot._nodes[nodes['children'][:, 0]],
                         self._ot._nodes[nodes['children'][:, 1]])
            self.face_proc_yz(self._ot._nodes[nodes['children'][:, 2]],
                         self._ot._nodes[nodes['children'][:, 3]])
            self.face_proc_yz(self._ot._nodes[nodes['children'][:, 4]],
                         self._ot._nodes[nodes['children'][:, 5]])
            self.face_proc_yz(self._ot._nodes[nodes['children'][:, 6]],
                         self._ot._nodes[nodes['children'][:, 7]])

            # Call self.edge_proc_x on the correct nodes
            self.edge_proc_x(self._ot._nodes[nodes['children'][:, 1]],
                        self._ot._nodes[nodes['children'][:, 5]],
                        self._ot._nodes[nodes['children'][:, 7]],
                        self._ot._nodes[nodes['children'][:, 3]])
            self.edge_proc_x(self._ot._nodes[nodes['children'][:, 0]],
                        self._ot._nodes[nodes['children'][:, 4]],
                        self._ot._nodes[nodes['children'][:, 6]],
                        self._ot._nodes[nodes['children'][:, 2]])

            # Call self.edge_proc_y on the correct nodes
            self.edge_proc_y(self._ot._nodes[nodes['children'][:, 2]],
                        self._ot._nodes[nodes['children'][:, 3]],
                        self._ot._nodes[nodes['children'][:, 7]],
                        self._ot._nodes[nodes['children'][:, 6]])
            self.edge_proc_y(self._ot._nodes[nodes['children'][:, 0]],
                        self._ot._nodes[nodes['children'][:, 1]],
                        self._ot._nodes[nodes['children'][:, 5]],
                        self._ot._nodes[nodes['children'][:, 4]])

            # Call self.edge_proc_z on the correct nodes
            self.edge_proc_z(self._ot._nodes[nodes['children'][:, 0]],
                        self._ot._nodes[nodes['children'][:, 2]],
                        self._ot._nodes[nodes['children'][:, 3]],
                        self._ot._nodes[nodes['children'][:, 1]])
            self.edge_proc_z(self._ot._nodes[nodes['children'][:, 4]],
                        self._ot._nodes[nodes['children'][:, 6]],
                        self._ot._nodes[nodes['children'][:, 7]],
                        self._ot._nodes[nodes['children'][:, 5]])

            # Call self.vert_proc on nodes 0-7
            self.vert_proc(self._ot._nodes[nodes['children'][:, 0]],
                      self._ot._nodes[nodes['children'][:, 1]],
                      self._ot._nodes[nodes['children'][:, 2]],
                      self._ot._nodes[nodes['children'][:, 3]],
                      self._ot._nodes[nodes['children'][:, 4]],
                      self._ot._nodes[nodes['children'][:, 5]],
                      self._ot._nodes[nodes['children'][:, 6]],
                      self._ot._nodes[nodes['children'][:, 7]])

    def face_proc_xy(self, n0, n1):
        # Make sure at least one of these nodes is subdivided & nself._ot equal to the 0-node
        face_mask = ((np.sum(n0['children'], axis=1) > 0) & (
        n0['depth'] > 0)) | (
                    (np.sum(n1['children'], axis=1) > 0) & n1['depth'] > 0)
        n0 = n0[face_mask]
        n1 = n1[face_mask]

        # Check 0-nodes
        n0, n1 = self.position_empty_node(n0, n1, [0, 0, 1])

        # Initialize resulting nodes to current nodes
        c4 = deepcopy(n0)
        c5 = deepcopy(n0)
        c6 = deepcopy(n0)
        c7 = deepcopy(n0)
        c0 = deepcopy(n1)
        c1 = deepcopy(n1)
        c2 = deepcopy(n1)
        c3 = deepcopy(n1)

        # Replace current nodes with their ordered children if present
        n0_subdivided = np.sum(n0['children'], axis=1) > 0
        n1_subdivided = np.sum(n1['children'], axis=1) > 0
        
        # DB: probably need to deal with missing nodes / children here too.

        if np.sum(n0_subdivided) > 0:
            c4[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 0]]
            c5[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 1]]
            c6[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 2]]
            c7[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 3]]

        if np.sum(n1_subdivided) > 0:
            c0[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 4]]
            c1[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 5]]
            c2[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 6]]
            c3[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 7]]

        if np.sum(n0_subdivided) > 0 | np.sum(n0_subdivided) > 0:
            # Call self.face_proc_xy, self.edge_proc_x, self.edge_proc_y, and self.vert_proc on resulting nodes
            self.face_proc_xy(c0, c4)
            self.face_proc_xy(c2, c6)
            self.face_proc_xy(c1, c5)
            self.face_proc_xy(c3, c7)

            self.edge_proc_x(c1, c5, c7, c3)
            self.edge_proc_x(c0, c4, c6, c2)

            self.edge_proc_y(c2, c3, c7, c6)
            self.edge_proc_y(c0, c1, c5, c4)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def face_proc_xz(self, n0, n1):
        # Make sure at least one of these nodes is subdivided & nself._ot equal to the 0-node
        face_mask = ((np.sum(n0['children'], axis=1) > 0) & (
        n0['depth'] > 0)) | (
                    (np.sum(n1['children'], axis=1) > 0) & n1['depth'] > 0)
        n0 = n0[face_mask]
        n1 = n1[face_mask]

        # Check 0-nodes
        n0, n1 = self.position_empty_node(n0, n1, [0, 1, 0])

        # Initialize resulting nodes to current nodes
        c2 = deepcopy(n0)
        c3 = deepcopy(n0)
        c6 = deepcopy(n0)
        c7 = deepcopy(n0)
        c0 = deepcopy(n1)
        c1 = deepcopy(n1)
        c4 = deepcopy(n1)
        c5 = deepcopy(n1)

        # Replace current nodes with their ordered children if present
        n0_subdivided = np.sum(n0['children'], axis=1) > 0
        n1_subdivided = np.sum(n1['children'], axis=1) > 0

        if np.sum(n0_subdivided) > 0:
            c2[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 0]]
            c3[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 1]]
            c6[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 4]]
            c7[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 5]]

        if np.sum(n1_subdivided) > 0:
            c0[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 2]]
            c1[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 3]]
            c4[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 6]]
            c5[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 7]]

        if np.sum(n0_subdivided) > 0 | np.sum(n0_subdivided) > 0:
            # Call self.face_proc_xy, self.edge_proc_x, self.edge_proc_y, and self.vert_proc on resulting nodes
            self.face_proc_xz(c0, c2)
            self.face_proc_xz(c1, c3)
            self.face_proc_xz(c4, c6)
            self.face_proc_xz(c5, c7)

            self.edge_proc_x(c1, c5, c7, c3)
            self.edge_proc_x(c0, c4, c6, c2)

            self.edge_proc_z(c0, c2, c3, c1)
            self.edge_proc_z(c4, c6, c7, c5)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def face_proc_yz(self, n0, n1):
        # Make sure at least one of these nodes is subdivided & nself._ot equal to the 0-node
        face_mask = ((np.sum(n0['children'], axis=1) > 0) & (
        n0['depth'] > 0)) | (
                    (np.sum(n1['children'], axis=1) > 0) & n1['depth'] > 0)
        n0 = n0[face_mask]
        n1 = n1[face_mask]

        # Check 0-nodes
        n0, n1 = self.position_empty_node(n0, n1, [1, 0, 0])

        # Initialize resulting nodes to current nodes
        c1 = deepcopy(n0)
        c3 = deepcopy(n0)
        c5 = deepcopy(n0)
        c7 = deepcopy(n0)
        c0 = deepcopy(n1)
        c2 = deepcopy(n1)
        c4 = deepcopy(n1)
        c6 = deepcopy(n1)

        # Replace current nodes with their ordered children if present
        n0_subdivided = np.sum(n0['children'], axis=1) > 0
        n1_subdivided = np.sum(n1['children'], axis=1) > 0

        if np.sum(n0_subdivided) > 0:
            c1[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 0]]
            c3[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 2]]
            c5[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 4]]
            c7[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 6]]

        if np.sum(n1_subdivided) > 0:
            c0[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 1]]
            c2[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 3]]
            c4[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 5]]
            c6[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 7]]

        if np.sum(n0_subdivided) > 0 | np.sum(n0_subdivided) > 0:
            # Call self.face_proc_xy, self.edge_proc_x, self.edge_proc_y, and self.vert_proc on resulting nodes
            self.face_proc_yz(c0, c1)
            self.face_proc_yz(c2, c3)
            self.face_proc_yz(c4, c5)
            self.face_proc_yz(c6, c7)

            self.edge_proc_y(c2, c3, c7, c6)
            self.edge_proc_y(c0, c1, c5, c4)

            self.edge_proc_z(c0, c2, c3, c1)
            self.edge_proc_z(c4, c6, c7, c5)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def edge_proc_x(self, n0, n1, n2, n3):
        # Make sure at least one of these nodes is subdivided & nself._ot equal to the 0-node
        face_mask = ((np.sum(n0['children'], axis=1) > 0) & (
        n0['depth'] > 0)) | (
                    (np.sum(n1['children'], axis=1) > 0) & (n1['depth'] > 0)) | \
                    ((np.sum(n2['children'], axis=1) > 0) & (
                    n2['depth'] > 0)) | (
                    (np.sum(n3['children'], axis=1) > 0) & (n3['depth'] > 0))
        n0 = n0[face_mask]
        n1 = n1[face_mask]
        n2 = n2[face_mask]
        n3 = n3[face_mask]

        # Check 0-nodes
        n0, n1 = self.position_empty_node(n0, n1, [0, 0, 1])
        n0, n2 = self.position_empty_node(n0, n2, [0, 1, 1])
        n0, n3 = self.position_empty_node(n0, n3, [0, 1, 0])
        n1, n2 = self.position_empty_node(n1, n2, [0, 1, 0])
        n1, n3 = self.position_empty_node(n1, n3, [0, 1, -1])
        n2, n3 = self.position_empty_node(n2, n3, [0, 0, -1])

        # Initialize resulting nodes to current nodes
        c6 = deepcopy(n0)
        c7 = deepcopy(n0)
        c2 = deepcopy(n1)
        c3 = deepcopy(n1)
        c0 = deepcopy(n2)
        c1 = deepcopy(n2)
        c4 = deepcopy(n3)
        c5 = deepcopy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided = np.sum(n0['children'], axis=1) > 0
        n1_subdivided = np.sum(n1['children'], axis=1) > 0
        n2_subdivided = np.sum(n2['children'], axis=1) > 0
        n3_subdivided = np.sum(n3['children'], axis=1) > 0

        if np.sum(n0_subdivided) > 0:
            c6[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 0]]
            c7[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 1]]

        if np.sum(n1_subdivided) > 0:
            c2[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 4]]
            c3[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 5]]

        if np.sum(n2_subdivided) > 0:
            c0[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 6]]
            c1[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 7]]

        if np.sum(n3_subdivided) > 0:
            c4[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 2]]
            c5[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 3]]

        if np.sum(n0_subdivided) > 0 | np.sum(n1_subdivided) > 0 | np.sum(
                n2_subdivided) > 0 | np.sum(n3_subdivided) > 0:
            self.edge_proc_x(c1, c5, c7, c3)
            self.edge_proc_x(c0, c4, c6, c2)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def edge_proc_y(self, n0, n1, n2, n3):
        # Make sure at least one of these nodes is subdivided & nself._ot equal to the 0-node
        face_mask = ((np.sum(n0['children'], axis=1) > 0) & (
        n0['depth'] > 0)) | (
                    (np.sum(n1['children'], axis=1) > 0) & (n1['depth'] > 0)) | \
                    ((np.sum(n2['children'], axis=1) > 0) & (
                    n2['depth'] > 0)) | (
                    (np.sum(n3['children'], axis=1) > 0) & (n3['depth'] > 0))
        n0 = n0[face_mask]
        n1 = n1[face_mask]
        n2 = n2[face_mask]
        n3 = n3[face_mask]

        # Check 0-nodes
        n0, n1 = self.position_empty_node(n0, n1, [1, 0, 0])
        n0, n2 = self.position_empty_node(n0, n2, [1, 0, 1])
        n0, n3 = self.position_empty_node(n0, n3, [0, 0, 1])
        n1, n2 = self.position_empty_node(n1, n2, [0, 0, 1])
        n1, n3 = self.position_empty_node(n1, n3, [-1, 0, 1])
        n2, n3 = self.position_empty_node(n2, n3, [-1, 0, 0])

        # Initialize resulting nodes to current nodes
        c5 = deepcopy(n0)
        c7 = deepcopy(n0)
        c4 = deepcopy(n1)
        c6 = deepcopy(n1)
        c0 = deepcopy(n2)
        c2 = deepcopy(n2)
        c1 = deepcopy(n3)
        c3 = deepcopy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided = np.sum(n0['children'], axis=1) > 0
        n1_subdivided = np.sum(n1['children'], axis=1) > 0
        n2_subdivided = np.sum(n2['children'], axis=1) > 0
        n3_subdivided = np.sum(n3['children'], axis=1) > 0

        if np.sum(n0_subdivided) > 0:
            c5[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 0]]
            c7[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 2]]

        if np.sum(n1_subdivided) > 0:
            c4[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 1]]
            c6[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 3]]

        if np.sum(n2_subdivided) > 0:
            c0[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 5]]
            c2[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 7]]

        if np.sum(n3_subdivided) > 0:
            c1[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 4]]
            c3[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 6]]

        if np.sum(n0_subdivided) > 0 | np.sum(n1_subdivided) > 0 | np.sum(
                n2_subdivided) > 0 | np.sum(n3_subdivided) > 0:
            self.edge_proc_y(c2, c3, c7, c6)
            self.edge_proc_y(c0, c1, c5, c4)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def edge_proc_z(self, n0, n1, n2, n3):
        # Make sure at least one of these nodes is subdivided & nself._ot equal to the 0-node
        face_mask = ((np.sum(n0['children'], axis=1) > 0) & (
        n0['depth'] > 0)) | (
                    (np.sum(n1['children'], axis=1) > 0) & (n1['depth'] > 0)) | \
                    ((np.sum(n2['children'], axis=1) > 0) & (
                    n2['depth'] > 0)) | (
                    (np.sum(n3['children'], axis=1) > 0) & (n3['depth'] > 0))
        n0 = n0[face_mask]
        n1 = n1[face_mask]
        n2 = n2[face_mask]
        n3 = n3[face_mask]

        # Check 0-nodes
        n0, n1 = self.position_empty_node(n0, n1, [0, 1, 0])
        n0, n2 = self.position_empty_node(n0, n2, [1, 1, 0])
        n0, n3 = self.position_empty_node(n0, n3, [1, 0, 0])
        n1, n2 = self.position_empty_node(n1, n2, [1, 0, 0])
        n1, n3 = self.position_empty_node(n1, n3, [1, -1, 0])
        n2, n3 = self.position_empty_node(n2, n3, [0, -1, 0])

        # Initialize resulting nodes to current nodes
        c3 = deepcopy(n0)
        c7 = deepcopy(n0)
        c1 = deepcopy(n1)
        c5 = deepcopy(n1)
        c0 = deepcopy(n2)
        c4 = deepcopy(n2)
        c2 = deepcopy(n3)
        c6 = deepcopy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided = np.sum(n0['children'], axis=1) > 0
        n1_subdivided = np.sum(n1['children'], axis=1) > 0
        n2_subdivided = np.sum(n2['children'], axis=1) > 0
        n3_subdivided = np.sum(n3['children'], axis=1) > 0

        if np.sum(n0_subdivided) > 0:
            c3[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 0]]
            c7[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 4]]

        if np.sum(n1_subdivided) > 0:
            c1[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 2]]
            c5[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 6]]

        if np.sum(n2_subdivided) > 0:
            c0[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 3]]
            c4[n0_subdivided] = self._ot._nodes[n0[n0_subdivided]['children'][:, 7]]

        if np.sum(n3_subdivided) > 0:
            c2[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 1]]
            c6[n1_subdivided] = self._ot._nodes[n1[n1_subdivided]['children'][:, 5]]

        if np.sum(n0_subdivided) > 0 | np.sum(n1_subdivided) > 0 | np.sum(
                n2_subdivided) > 0 | np.sum(n3_subdivided) > 0:
            self.edge_proc_z(c0, c2, c3, c1)
            self.edge_proc_z(c4, c6, c7, c5)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def vert_proc(self, n0, n1, n2, n3, n4, n5, n6, n7):
        leaf_nodes = (np.sum(n0['children'], axis=1) == 0) & (
        np.sum(n1['children'], axis=1) == 0) & \
                     (np.sum(n2['children'], axis=1) == 0) & (
                     np.sum(n3['children'], axis=1) == 0) & \
                     (np.sum(n4['children'], axis=1) == 0) & (
                     np.sum(n5['children'], axis=1) == 0) & \
                     (np.sum(n6['children'], axis=1) == 0) & (
                     np.sum(n7['children'], axis=1) == 0)

        if np.sum(leaf_nodes) > 0:
            inds = np.where(leaf_nodes)
            self.vertices.append(
                np.swapaxes(np.array([n0[inds]['centre'], n1[inds]['centre'], \
                                      n3[inds]['centre'], n2[inds]['centre'], \
                                      n4[inds]['centre'], n5[inds]['centre'], \
                                      n7[inds]['centre'], n6[inds]['centre']]),
                            0, 1))
            self.values.append(
                np.swapaxes(np.array([n0[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n0[inds][
                                                                  'depth']),
                                              axis=0),
                                     n1[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n1[inds][
                                                                  'depth']),
                                              axis=0),
                                     n3[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n2[inds][
                                                                  'depth']),
                                              axis=0),
                                     n2[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n3[inds][
                                                                  'depth']),
                                              axis=0),
                                     n4[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n4[inds][
                                                                  'depth']),
                                              axis=0),
                                     n5[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n5[inds][
                                                                  'depth']),
                                              axis=0),
                                     n7[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n6[inds][
                                                                  'depth']),
                                              axis=0),
                                     n6[inds]['nPoints'] /
                                      np.prod(self._ot.box_size(n7[inds][
                                                                'depth']),
                                              axis=0)]),
                            0, 1))

        if np.sum(~leaf_nodes) > 0:

            # Check 0-nodes
            n0, n1 = self.position_empty_node(n0, n1, [1, 0, 0])
            n0, n2 = self.position_empty_node(n0, n2, [0, 1, 0])
            n0, n3 = self.position_empty_node(n0, n3, [1, 1, 0])
            n0, n4 = self.position_empty_node(n0, n4, [0, 0, 1])
            n0, n5 = self.position_empty_node(n0, n5, [1, 0, 1])
            n0, n6 = self.position_empty_node(n0, n6, [0, 1, 1])
            n0, n7 = self.position_empty_node(n0, n7, [1, 1, 1])
            n1, n2 = self.position_empty_node(n1, n2, [-1, 1, 0])
            n1, n3 = self.position_empty_node(n1, n3, [0, 1, 0])
            n1, n4 = self.position_empty_node(n1, n4, [-1, 0, 1])
            n1, n5 = self.position_empty_node(n1, n5, [0, 0, 1])
            n1, n6 = self.position_empty_node(n1, n6, [-1, 1, 1])
            n1, n7 = self.position_empty_node(n1, n7, [0, 1, 1])
            n2, n3 = self.position_empty_node(n2, n3, [1, 0, 0])
            n2, n4 = self.position_empty_node(n2, n4, [0, -1, 1])
            n2, n5 = self.position_empty_node(n2, n5, [1, -1, 1])
            n2, n6 = self.position_empty_node(n2, n6, [0, 0, 1])
            n2, n7 = self.position_empty_node(n2, n7, [1, 0, 1])
            n3, n4 = self.position_empty_node(n3, n4, [-1, -1, 1])
            n3, n5 = self.position_empty_node(n3, n5, [0, -1, 1])
            n3, n6 = self.position_empty_node(n3, n6, [-1, 0, 1])
            n3, n7 = self.position_empty_node(n3, n7, [0, 0, 1])
            n4, n5 = self.position_empty_node(n4, n5, [1, 0, 0])
            n4, n6 = self.position_empty_node(n4, n6, [0, 1, 0])
            n4, n7 = self.position_empty_node(n4, n7, [1, 1, 0])
            n5, n6 = self.position_empty_node(n5, n6, [-1, 1, 0])
            n5, n7 = self.position_empty_node(n5, n7, [0, 1, 0])
            n6, n7 = self.position_empty_node(n6, n7, [1, 0, 0])

            # Re-check leaf nodes
            new_leaf_nodes = (np.sum(n0['children'], axis=1) == 0) & \
                (np.sum(n1['children'], axis=1) == 0) & \
                (np.sum(n2['children'], axis=1) == 0) & \
                (np.sum(n3['children'], axis=1) == 0) & \
                (np.sum(n4['children'], axis=1) == 0) & \
                (np.sum(n5['children'], axis=1) == 0) & \
                (np.sum(n6['children'], axis=1) == 0) & \
                (np.sum(n7['children'], axis=1) == 0)
            if np.sum(new_leaf_nodes) > 0:
                new_inds = np.where(new_leaf_nodes)
                self.vert_proc(n0[new_inds], n1[new_inds], n2[new_inds],
                          n3[new_inds], n4[new_inds], n5[new_inds],
                          n6[new_inds], n7[new_inds])

            # Initialize resulting nodes to current nodes
            c0 = deepcopy(n0)
            c1 = deepcopy(n1)
            c2 = deepcopy(n2)
            c3 = deepcopy(n3)
            c4 = deepcopy(n4)
            c5 = deepcopy(n5)
            c6 = deepcopy(n6)
            c7 = deepcopy(n7)

            inds = np.where(~new_leaf_nodes)

            # Replace current nodes with their ordered children if present
            n0_subdivided = np.sum(n0[inds]['children'], axis=1) > 0
            n1_subdivided = np.sum(n1[inds]['children'], axis=1) > 0
            n2_subdivided = np.sum(n2[inds]['children'], axis=1) > 0
            n3_subdivided = np.sum(n3[inds]['children'], axis=1) > 0
            n4_subdivided = np.sum(n4[inds]['children'], axis=1) > 0
            n5_subdivided = np.sum(n5[inds]['children'], axis=1) > 0
            n6_subdivided = np.sum(n6[inds]['children'], axis=1) > 0
            n7_subdivided = np.sum(n7[inds]['children'], axis=1) > 0

            if np.sum(n0_subdivided) > 0:
                c0[inds][n0_subdivided] = self._ot._nodes[
                    n0[inds][n0_subdivided]['children'][:, 7]]
            if np.sum(n1_subdivided) > 0:
                c1[inds][n1_subdivided] = self._ot._nodes[
                    n1[inds][n1_subdivided]['children'][:, 6]]
            if np.sum(n2_subdivided) > 0:
                c2[inds][n2_subdivided] = self._ot._nodes[
                    n2[inds][n2_subdivided]['children'][:, 5]]
            if np.sum(n3_subdivided) > 0:
                c3[inds][n3_subdivided] = self._ot._nodes[
                    n3[inds][n3_subdivided]['children'][:, 4]]
            if np.sum(n4_subdivided) > 0:
                c4[inds][n4_subdivided] = self._ot._nodes[
                    n4[inds][n4_subdivided]['children'][:, 3]]
            if np.sum(n5_subdivided) > 0:
                c5[inds][n5_subdivided] = self._ot._nodes[
                    n5[inds][n5_subdivided]['children'][:, 2]]
            if np.sum(n6_subdivided) > 0:
                c6[inds][n6_subdivided] = self._ot._nodes[
                    n6[inds][n6_subdivided]['children'][:, 1]]
            if np.sum(n7_subdivided) > 0:
                c7[inds][n7_subdivided] = self._ot._nodes[
                    n7[inds][n7_subdivided]['children'][:, 0]]

            if np.sum(n0_subdivided) > 0 | np.sum(n1_subdivided) > 0 | np.sum(
                    n2_subdivided) > 0 | np.sum(n3_subdivided) > 0 | np.sum(
                    n4_subdivided) > 0 | np.sum(n5_subdivided) > 0 | np.sum(
                    n6_subdivided) > 0 | np.sum(n7_subdivided) > 0:
                self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)