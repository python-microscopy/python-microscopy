import numpy as np
from copy import deepcopy

from PYME.experimental.modified_marching_cubes import ModifiedMarchingCubes


class DualMarchingCubes(ModifiedMarchingCubes):
    def __init__(self, isolevel=0):
        super(DualMarchingCubes, self).__init__(isolevel)
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

    def march(self, return_triangles=True, dual_march=True):
        return super(DualMarchingCubes, self).march(return_triangles, dual_march)

    def position_empty_node(self, n0, n1, shift):
        """
        Function that considers nodes pairwise to replace 0-node if one of the
        nodes is the 0-node and the self._other is nself._ot.

        Parameters
        ----------
        n0 : _octree._node
            A node in the same subdivision as n1
        n1 : _octree._node
            A node in the same subdivision as n0
        shift : list
            List of x, y, z representing unit vector to move from n0 to n1 in
            3-space
        """
        # We need to replace any instances of the 0-node with an empty node
        # corresponding to a position directly across from the self._other
        # node. Essentially, we rebuild this portion of the octree.
        n0_root_mask = (n0['depth'] == 0) #& ((n1['children']).sum(1) > 0)
        n1_root_mask = (n1['depth'] == 0) #& ((n0['children']).sum(1) > 0)

        if (np.sum(n0_root_mask) > 0):
            inds = np.where(n0_root_mask)
            empty_node = np.zeros_like(n0[inds])
            # empty_node['nPoints'] = 0
            empty_node['depth'] = n1[inds]['depth']
            empty_node['centre'] = n1[inds]['centre'] + np.vstack(
                self._ot.box_size(n1[inds]['depth'])).T * np.array(shift) * -1
            n0[inds] = empty_node

        if (np.sum(n1_root_mask) > 0):
            inds = np.where(n1_root_mask)
            empty_node = np.zeros_like(n1[inds])
            # empty_node['nPoints'] = 0
            empty_node['depth'] = n0[inds]['depth']
            empty_node['centre'] = n0[inds]['centre'] + np.vstack(
                self._ot.box_size(n0[inds]['depth'])).T * np.array(shift)
            n1[inds] = empty_node
            
        return n0, n1

    def subdivided(self, nodes):
        """ 
        Returns whether or not any of the nodes are subdivided and,
        if so, which ones are.

        Parameters
        ----------
            nodes : np.array
                Node or nodes of an octree.
        
        Returns
        -------
            divisions : np.array
                True/False array indicating whether or not each node in
                nodes is subdivided.
            is_subdivided : np.array
                Are of the nodes in nodes subdivided?
        """
        divisions = (np.sum(nodes['children'], axis=1) > 0)
        is_subdivided = (np.sum(divisions) > 0)

        return divisions, is_subdivided

    def update_subdivision(self, node):
        """
        Give non-root-node options for all children of a node that is subdivided.
        This is meant to recover empty node positions on a sparse octree.
        """
        
        # Grab the children
        n0 = np.copy(self._ot._nodes[node['children'][:, 0]])
        n1 = np.copy(self._ot._nodes[node['children'][:, 1]])
        n2 = np.copy(self._ot._nodes[node['children'][:, 2]])
        n3 = np.copy(self._ot._nodes[node['children'][:, 3]])
        n4 = np.copy(self._ot._nodes[node['children'][:, 4]])
        n5 = np.copy(self._ot._nodes[node['children'][:, 5]])
        n6 = np.copy(self._ot._nodes[node['children'][:, 6]])
        n7 = np.copy(self._ot._nodes[node['children'][:, 7]])

        # Make sure the zero nodes are set to a terminal node
        # at the correct spatial position.
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

        # Return the subdivided nodes
        return n0, n1, n2, n3, n4, n5, n6, n7

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
            # children = self._ot._nodes[nodes['children'][nodes['children'] > 0]]

            n0, n1, n2, n3, n4, n5, n6, n7 = self.update_subdivision(nodes)

            children = np.hstack((n0, n1, n2, n3, n4, n5, n6, n7))

            self.node_proc(children)

            # We call face_proc_<plane> and edge_proc_<plane> on nodes
            # in a specific orderso we can track where subdivided nodes will be
            # in reference to adjacent subdivided nodes
            
            # DB: do we need to worry about sparse children for all of these calls? 
            # At present you are any non-occupied nodes will be interpreted as the root node

            # Call self.face_proc_xy on the correct nodes
            self.face_proc_xy(n0, n4)
            self.face_proc_xy(n2, n6)
            self.face_proc_xy(n1, n5)
            self.face_proc_xy(n3, n7)

            self.face_proc_xz(n0, n2)
            self.face_proc_xz(n1, n3)
            self.face_proc_xz(n4, n6)
            self.face_proc_xz(n5, n7)

            self.face_proc_yz(n0, n1)
            self.face_proc_yz(n2, n3)
            self.face_proc_yz(n4, n5)
            self.face_proc_yz(n6, n7)

            self.edge_proc_x(n1, n5, n7, n3)
            self.edge_proc_x(n0, n4, n6, n2)

            self.edge_proc_y(n2, n3, n7, n6)
            self.edge_proc_y(n0, n1, n5, n4)

            self.edge_proc_z(n0, n2, n3, n1)
            self.edge_proc_z(n4, n6, n7, n5)

            self.vert_proc(n0, n1, n2, n3, n4, n5, n6, n7)

    def face_proc_xy(self, n0, n1):

        # Initialize resulting nodes to current nodes
        c0 = np.copy(n0)
        c1 = np.copy(n0)
        c2 = np.copy(n0)
        c3 = np.copy(n0)
        c4 = np.copy(n1)
        c5 = np.copy(n1)
        c6 = np.copy(n1)
        c7 = np.copy(n1)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)

        if is_n0_subdivided:
            
            _, _, _, _, u4, u5, u6, u7 = self.update_subdivision(n0[n0_subdivided])

            c0[n0_subdivided] = u4
            c1[n0_subdivided] = u5
            c2[n0_subdivided] = u6
            c3[n0_subdivided] = u7

        if is_n1_subdivided:

            u0, u1, u2, u3, _, _, _, _ = self.update_subdivision(n1[n1_subdivided])

            c4[n1_subdivided] = u0
            c5[n1_subdivided] = u1
            c6[n1_subdivided] = u2
            c7[n1_subdivided] = u3

        if is_n0_subdivided or is_n1_subdivided:
            # Call self.face_proc_xy, self.edge_proc_x, self.edge_proc_y, and
            # self.vert_proc on resulting nodes

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

        # Initialize resulting nodes to current nodes
        c0 = np.copy(n0)
        c1 = np.copy(n0)
        c4 = np.copy(n0)
        c5 = np.copy(n0)
        c2 = np.copy(n1)
        c3 = np.copy(n1)
        c6 = np.copy(n1)
        c7 = np.copy(n1)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)

        if is_n0_subdivided:
            
            _, _, u2, u3, _, _, u6, u7 = self.update_subdivision(n0[n0_subdivided])

            c0[n0_subdivided] = u2
            c1[n0_subdivided] = u3
            c4[n0_subdivided] = u6
            c5[n0_subdivided] = u7

        if is_n1_subdivided:

            u0, u1, _, _, u4, u5, _, _ = self.update_subdivision(n1[n1_subdivided])

            c2[n1_subdivided] = u0
            c3[n1_subdivided] = u1
            c6[n1_subdivided] = u4
            c7[n1_subdivided] = u5
            
        if is_n0_subdivided or is_n1_subdivided:
            # Call self.face_proc_xy, self.edge_proc_x, self.edge_proc_y, and
            # self.vert_proc on resulting nodes
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

        # Initialize resulting nodes to current nodes
        c0 = np.copy(n0)
        c2 = np.copy(n0)
        c4 = np.copy(n0)
        c6 = np.copy(n0)
        c1 = np.copy(n1)
        c3 = np.copy(n1)
        c5 = np.copy(n1)
        c7 = np.copy(n1)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)

        if is_n0_subdivided:
            
            _, u1, _, u3, _, u5, _, u7 = self.update_subdivision(n0[n0_subdivided])

            c0[n0_subdivided] = u1
            c2[n0_subdivided] = u3
            c4[n0_subdivided] = u5
            c6[n0_subdivided] = u7

        if is_n1_subdivided:

            u0, _, u2, _, u4, _, u6, _ = self.update_subdivision(n1[n1_subdivided])

            c1[n1_subdivided] = u0
            c3[n1_subdivided] = u2
            c5[n1_subdivided] = u4
            c7[n1_subdivided] = u6

        if is_n0_subdivided or is_n1_subdivided:
            # Call self.face_proc_xy, self.edge_proc_x, self.edge_proc_y, and
            # self.vert_proc on resulting nodes
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

        # Initialize resulting nodes to current nodes
        c0 = np.copy(n0)
        c1 = np.copy(n0)
        c4 = np.copy(n1)
        c5 = np.copy(n1)
        c6 = np.copy(n2)
        c7 = np.copy(n2)
        c2 = np.copy(n3)
        c3 = np.copy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)
        n2_subdivided, is_n2_subdivided = self.subdivided(n2)
        n3_subdivided, is_n3_subdivided = self.subdivided(n3)

        if is_n2_subdivided:
            u0, u1, _, _, _, _, _, _ = self.update_subdivision(n2[n2_subdivided])

            c6[n2_subdivided] = u0
            c7[n2_subdivided] = u1

        if is_n3_subdivided:
            _, _, _, _, u4, u5, _, _ = self.update_subdivision(n3[n3_subdivided])
            
            c2[n3_subdivided] = u4
            c3[n3_subdivided] = u5

        if is_n0_subdivided:
            _, _, _, _, _, _, u6, u7 = self.update_subdivision(n0[n0_subdivided])

            c0[n0_subdivided] = u6
            c1[n0_subdivided] = u7

        if is_n1_subdivided:
            _, _, u2, u3, _, _, _, _ = self.update_subdivision(n1[n1_subdivided])

            c4[n1_subdivided] = u2
            c5[n1_subdivided] = u3

        if is_n0_subdivided or is_n1_subdivided or is_n2_subdivided or is_n3_subdivided:
            self.edge_proc_x(c1, c5, c7, c3)
            self.edge_proc_x(c0, c4, c6, c2)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def edge_proc_y(self, n0, n1, n2, n3):

        # Initialize resulting nodes to current nodes
        c0 = np.copy(n0)
        c2 = np.copy(n0)
        c1 = np.copy(n1)
        c3 = np.copy(n1)
        c5 = np.copy(n2)
        c7 = np.copy(n2)
        c4 = np.copy(n3)
        c6 = np.copy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)
        n2_subdivided, is_n2_subdivided = self.subdivided(n2)
        n3_subdivided, is_n3_subdivided = self.subdivided(n3)

        if is_n2_subdivided:

            u0, _, u2, _, _, _, _, _ = self.update_subdivision(n2[n2_subdivided])

            c5[n2_subdivided] = u0
            c7[n2_subdivided] = u2

        if is_n3_subdivided:

            _, u1, _, u3, _, _, _, _ = self.update_subdivision(n3[n3_subdivided])

            c4[n3_subdivided] = u1
            c6[n3_subdivided] = u3

        if is_n0_subdivided:

            _, _, _, _, _, u5, _, u7 = self.update_subdivision(n0[n0_subdivided])

            c0[n0_subdivided] = u5
            c2[n0_subdivided] = u7


        if is_n1_subdivided:

            _, _, _, _, u4, _, u6, _ = self.update_subdivision(n1[n1_subdivided])

            c1[n1_subdivided] = u4
            c3[n1_subdivided] = u6

        if is_n0_subdivided or is_n1_subdivided or is_n2_subdivided or is_n3_subdivided:
            self.edge_proc_y(c2, c3, c7, c6)
            self.edge_proc_y(c0, c1, c5, c4)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def edge_proc_z(self, n0, n1, n2, n3):

        # Initialize resulting nodes to current nodes
        c0 = np.copy(n0)
        c4 = np.copy(n0)
        c2 = np.copy(n1)
        c6 = np.copy(n1)
        c3 = np.copy(n2)
        c7 = np.copy(n2)
        c1 = np.copy(n3)
        c5 = np.copy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)
        n2_subdivided, is_n2_subdivided = self.subdivided(n2)
        n3_subdivided, is_n3_subdivided = self.subdivided(n3)

        if is_n2_subdivided:

            u0, _, _, _, u4, _, _, _ = self.update_subdivision(n2[n2_subdivided])

            c3[n2_subdivided] = u0
            c7[n2_subdivided] = u4


        if is_n3_subdivided:

            _, _, u2, _, _, _, u6, _ = self.update_subdivision(n3[n3_subdivided])
            
            c1[n3_subdivided] = u2
            c5[n3_subdivided] = u6

        if is_n0_subdivided:

            _, _, _, u3, _, _, _, u7 = self.update_subdivision(n0[n0_subdivided])

            c0[n0_subdivided] = u3
            c4[n0_subdivided] = u7

        if (np.sum(n1_subdivided) > 0):

            _, u1, _, _, _, u5, _, _ = self.update_subdivision(n1[n1_subdivided])

            c2[n1_subdivided] = u1
            c6[n1_subdivided] = u5

        if is_n0_subdivided or is_n1_subdivided or is_n2_subdivided or is_n3_subdivided:
            self.edge_proc_z(c4, c6, c7, c5)
            self.edge_proc_z(c0, c2, c3, c1)

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

        if (np.sum(leaf_nodes) > 0):
            inds = np.where(leaf_nodes)
            self.vertices.append(
                np.swapaxes(np.array([n0[inds]['centre'], n1[inds]['centre'], \
                                      n2[inds]['centre'], n3[inds]['centre'], \
                                      n4[inds]['centre'], n5[inds]['centre'], \
                                      n6[inds]['centre'], n7[inds]['centre']]),
                            0, 1))
            self.values.append(
                np.swapaxes(np.array([n0[inds]['nPoints'] / np.prod(self._ot.box_size(n0[inds]['depth']), axis=0),
                                     n1[inds]['nPoints'] / np.prod(self._ot.box_size(n1[inds]['depth']), axis=0),
                                     n2[inds]['nPoints'] / np.prod(self._ot.box_size(n3[inds]['depth']), axis=0),
                                     n3[inds]['nPoints'] / np.prod(self._ot.box_size(n2[inds]['depth']), axis=0),
                                     n4[inds]['nPoints'] / np.prod(self._ot.box_size(n4[inds]['depth']), axis=0),
                                     n5[inds]['nPoints'] / np.prod(self._ot.box_size(n5[inds]['depth']), axis=0),
                                     n6[inds]['nPoints'] / np.prod(self._ot.box_size(n7[inds]['depth']), axis=0),
                                     n7[inds]['nPoints'] / np.prod(self._ot.box_size(n6[inds]['depth']), axis=0)]),
                            0, 1))

        if np.sum(~(leaf_nodes).astype(bool)) > 0:

            inds = np.where(~(leaf_nodes).astype(bool))

            # Initialize resulting nodes to current nodes
            c0 = np.copy(n0[inds])
            c1 = np.copy(n1[inds])
            c2 = np.copy(n2[inds])
            c3 = np.copy(n3[inds])
            c4 = np.copy(n4[inds])
            c5 = np.copy(n5[inds])
            c6 = np.copy(n6[inds])
            c7 = np.copy(n7[inds])

            # Replace current nodes with their ordered children if present
            c0_subdivided, is_c0_subdivided = self.subdivided(c0)
            c1_subdivided, is_c1_subdivided = self.subdivided(c1)
            c2_subdivided, is_c2_subdivided = self.subdivided(c2)
            c3_subdivided, is_c3_subdivided = self.subdivided(c3)
            c4_subdivided, is_c4_subdivided = self.subdivided(c4)
            c5_subdivided, is_c5_subdivided = self.subdivided(c5)
            c6_subdivided, is_c6_subdivided = self.subdivided(c6)
            c7_subdivided, is_c7_subdivided = self.subdivided(c7)

            if is_c0_subdivided:
                _, _, _, _, _, _, _, u7 = self.update_subdivision(c0[c0_subdivided])
                c0[c0_subdivided] = u7
            if is_c1_subdivided:
                _, _, _, _, _, _, u6, _ = self.update_subdivision(c1[c1_subdivided])
                c1[c1_subdivided] = u6
            if is_c2_subdivided:
                _, _, _, _, _, u5, _, _ = self.update_subdivision(c2[c2_subdivided])
                c2[c2_subdivided] = u5
            if is_c3_subdivided:
                _, _, _, _, u4, _, _, _ = self.update_subdivision(c3[c3_subdivided])
                c3[c3_subdivided] = u4
            if is_c4_subdivided:
                _, _, _, u3, _, _, _, _ = self.update_subdivision(c4[c4_subdivided])
                c4[c4_subdivided] = u3
            if is_c5_subdivided:
                _, _, u2, _, _, _, _, _ = self.update_subdivision(c5[c5_subdivided])
                c5[c5_subdivided] = u2
            if is_c6_subdivided:
                _, u1, _, _, _, _, _, _ = self.update_subdivision(c6[c6_subdivided])
                c6[c6_subdivided] = u1
            if is_c7_subdivided:
                u0, _, _, _, _, _, _, _ = self.update_subdivision(c7[c7_subdivided])
                c7[c7_subdivided] = u0

            if is_c0_subdivided or is_c1_subdivided or is_c2_subdivided or is_c3_subdivided or is_c4_subdivided or is_c5_subdivided or is_c6_subdivided or is_c7_subdivided:
                self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)
