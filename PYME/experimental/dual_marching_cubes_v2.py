import numpy as np
from copy import deepcopy

from PYME.experimental.modified_marching_cubes import ModifiedMarchingCubes#, PiecewiseLinearMMC


class DualMarchingCubes(ModifiedMarchingCubes):
    def __init__(self, isolevel=0):
        super(DualMarchingCubes, self).__init__(isolevel)
        self._ot = None

    def set_octree(self, ot):
        self._ot = ot  # Assign the octree

        # Make vertices/values a list instead of None
        self.vertices = []
        self.values = []
        self.depths = []
        
        #precalculate shift scales for empty boxes
        max_depth = self._ot._nodes['depth'].max()
        self._empty_shift_scales = 0.5*np.vstack(self._ot.box_size(np.arange(max_depth + 1))).T
        # precaluclate box sizes
        self._density_sc = 1.0/np.prod(self._ot.box_size(np.arange(max_depth + 1)), axis=0)
        
        #TODO - get this from the octree
        self._octant_sign = np.array([[2 * (n & 1) - 1, (n & 2) - 1, (n & 4) / 2 - 1] for n in range(8)])
        

        self.node_proc(self._ot._nodes[0])  # Create the dual grid

        # Make vertices/values np arrays
        self.vertices = np.vstack(self.vertices).astype('float64')
        self.values = np.vstack(self.values).astype('float64')
        self.depths = np.vstack(self.depths).astype('float64')

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
        
        #print n0.shape

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
    
    def _empty_node_v2(self, nj, parent, j):
        #print nj.shape, parent.shape
        n_root_mask = (nj['depth'] == 0)
        
        if np.any(n_root_mask):
            inds = np.where(n_root_mask)

            empty_node = np.zeros_like(nj[inds])
            
            empty_node['depth'] = parent[inds]['depth'] + 1
            empty_node['centre'] = parent[inds]['centre'] + self._empty_shift_scales[empty_node['depth'], :] * self._octant_sign[j, :][None,:]
        
            nj[inds] = empty_node
        return nj

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

    def update_subdivision(self, node, children=range(8)):
        """
        Give non-root-node options for all children of a node that is subdivided.
        This is meant to recover empty node positions on a sparse octree.
        """
        
        # # Grab the children
        # n0 = np.copy(self._ot._nodes[node['children'][:, 0]])
        # n1 = np.copy(self._ot._nodes[node['children'][:, 1]])
        # n2 = np.copy(self._ot._nodes[node['children'][:, 2]])
        # n3 = np.copy(self._ot._nodes[node['children'][:, 3]])
        # n4 = np.copy(self._ot._nodes[node['children'][:, 4]])
        # n5 = np.copy(self._ot._nodes[node['children'][:, 5]])
        # n6 = np.copy(self._ot._nodes[node['children'][:, 6]])
        # n7 = np.copy(self._ot._nodes[node['children'][:, 7]])
        #
        # # Make sure the zero nodes are set to a terminal node
        # # at the correct spatial position.
        # # n0, n1 = self.position_empty_node(n0, n1, [1, 0, 0])
        # # n0, n2 = self.position_empty_node(n0, n2, [0, 1, 0])
        # # n0, n3 = self.position_empty_node(n0, n3, [1, 1, 0])
        # # n0, n4 = self.position_empty_node(n0, n4, [0, 0, 1])
        # # n0, n5 = self.position_empty_node(n0, n5, [1, 0, 1])
        # # n0, n6 = self.position_empty_node(n0, n6, [0, 1, 1])
        # # n0, n7 = self.position_empty_node(n0, n7, [1, 1, 1])
        # # n1, n2 = self.position_empty_node(n1, n2, [-1, 1, 0])
        # # n1, n3 = self.position_empty_node(n1, n3, [0, 1, 0])
        # # n1, n4 = self.position_empty_node(n1, n4, [-1, 0, 1])
        # # n1, n5 = self.position_empty_node(n1, n5, [0, 0, 1])
        # # n1, n6 = self.position_empty_node(n1, n6, [-1, 1, 1])
        # # n1, n7 = self.position_empty_node(n1, n7, [0, 1, 1])
        # # n2, n3 = self.position_empty_node(n2, n3, [1, 0, 0])
        # # n2, n4 = self.position_empty_node(n2, n4, [0, -1, 1])
        # # n2, n5 = self.position_empty_node(n2, n5, [1, -1, 1])
        # # n2, n6 = self.position_empty_node(n2, n6, [0, 0, 1])
        # # n2, n7 = self.position_empty_node(n2, n7, [1, 0, 1])
        # # n3, n4 = self.position_empty_node(n3, n4, [-1, -1, 1])
        # # n3, n5 = self.position_empty_node(n3, n5, [0, -1, 1])
        # # n3, n6 = self.position_empty_node(n3, n6, [-1, 0, 1])
        # # n3, n7 = self.position_empty_node(n3, n7, [0, 0, 1])
        # # n4, n5 = self.position_empty_node(n4, n5, [1, 0, 0])
        # # n4, n6 = self.position_empty_node(n4, n6, [0, 1, 0])
        # # n4, n7 = self.position_empty_node(n4, n7, [1, 1, 0])
        # # n5, n6 = self.position_empty_node(n5, n6, [-1, 1, 0])
        # # n5, n7 = self.position_empty_node(n5, n7, [0, 1, 0])
        # # n6, n7 = self.position_empty_node(n6, n7, [1, 0, 0])
        #
        # n0 = self._empty_node_v2(n0, node, 0)
        # n1 = self._empty_node_v2(n1, node, 1)
        # n2 = self._empty_node_v2(n2, node, 2)
        # n3 = self._empty_node_v2(n3, node, 3)
        # n4 = self._empty_node_v2(n4, node, 4)
        # n5 = self._empty_node_v2(n5, node, 5)
        # n6 = self._empty_node_v2(n6, node, 6)
        # n7 = self._empty_node_v2(n7, node, 7)
        #
        # # Return the subdivided nodes
        # return n0, n1, n2, n3, n4, n5, n6, n7
    
        return [self._empty_node_v2(np.copy(self._ot._nodes[node['children'][:, j]]), node, j) for j in children]

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
        c0, c1, c2, c3 = np.copy(n0), np.copy(n0), np.copy(n0), np.copy(n0)
        c4, c5, c6, c7 = np.copy(n1), np.copy(n1), np.copy(n1), np.copy(n1)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)

        if is_n0_subdivided:
            u4, u5, u6, u7 = self.update_subdivision(n0[n0_subdivided], [4, 5, 6, 7])

            c0[n0_subdivided] = u4
            c1[n0_subdivided] = u5
            c2[n0_subdivided] = u6
            c3[n0_subdivided] = u7

        if is_n1_subdivided:
            u0, u1, u2, u3 = self.update_subdivision(n1[n1_subdivided], [0,1,2,3])

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
        c0, c1, c4, c5 = np.copy(n0), np.copy(n0), np.copy(n0), np.copy(n0)
        c2, c3, c6, c7 = np.copy(n1), np.copy(n1), np.copy(n1), np.copy(n1)
        

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)

        if is_n0_subdivided:
            u2, u3, u6, u7 = self.update_subdivision(n0[n0_subdivided], [2,3,6,7])

            c0[n0_subdivided] = u2
            c1[n0_subdivided] = u3
            c4[n0_subdivided] = u6
            c5[n0_subdivided] = u7

        if is_n1_subdivided:
            u0, u1, u4, u5 = self.update_subdivision(n1[n1_subdivided], [0,1,4,5])

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
        c0, c2, c4, c6 = np.copy(n0), np.copy(n0), np.copy(n0), np.copy(n0)
        c1, c3, c5, c7 = np.copy(n1), np.copy(n1), np.copy(n1), np.copy(n1)


        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)

        if is_n0_subdivided:
            
            u1, u3, u5, u7 = self.update_subdivision(n0[n0_subdivided], [1,3,5,7])

            c0[n0_subdivided] = u1
            c2[n0_subdivided] = u3
            c4[n0_subdivided] = u5
            c6[n0_subdivided] = u7

        if is_n1_subdivided:

            u0, u2, u4, u6 = self.update_subdivision(n1[n1_subdivided], [0,2,4,6])

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
        c0, c1 = np.copy(n0), np.copy(n0)
        c4, c5 = np.copy(n1), np.copy(n1)
        c6, c7 = np.copy(n2), np.copy(n2)
        c2, c3 = np.copy(n3), np.copy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)
        n2_subdivided, is_n2_subdivided = self.subdivided(n2)
        n3_subdivided, is_n3_subdivided = self.subdivided(n3)

        if is_n2_subdivided:
            u0, u1 = self.update_subdivision(n2[n2_subdivided], [0,1])

            c6[n2_subdivided] = u0
            c7[n2_subdivided] = u1

        if is_n3_subdivided:
            u4, u5 = self.update_subdivision(n3[n3_subdivided], [4,5])
            
            c2[n3_subdivided] = u4
            c3[n3_subdivided] = u5

        if is_n0_subdivided:
            u6, u7 = self.update_subdivision(n0[n0_subdivided], [6,7])

            c0[n0_subdivided] = u6
            c1[n0_subdivided] = u7

        if is_n1_subdivided:
            u2, u3 = self.update_subdivision(n1[n1_subdivided], [2,3])

            c4[n1_subdivided] = u2
            c5[n1_subdivided] = u3

        if is_n0_subdivided or is_n1_subdivided or is_n2_subdivided or is_n3_subdivided:
            self.edge_proc_x(c1, c5, c7, c3)
            self.edge_proc_x(c0, c4, c6, c2)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def edge_proc_y(self, n0, n1, n2, n3):

        # Initialize resulting nodes to current nodes
        c0, c2 = np.copy(n0), np.copy(n0)
        c1, c3 = np.copy(n1), np.copy(n1)
        c5, c7 = np.copy(n2), np.copy(n2)
        c4, c6 = np.copy(n3), np.copy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)
        n2_subdivided, is_n2_subdivided = self.subdivided(n2)
        n3_subdivided, is_n3_subdivided = self.subdivided(n3)

        if is_n2_subdivided:
            u0, u2 = self.update_subdivision(n2[n2_subdivided], [0,2])

            c5[n2_subdivided] = u0
            c7[n2_subdivided] = u2

        if is_n3_subdivided:
            u1, u3 = self.update_subdivision(n3[n3_subdivided], [1,3])

            c4[n3_subdivided] = u1
            c6[n3_subdivided] = u3

        if is_n0_subdivided:
            u5, u7 = self.update_subdivision(n0[n0_subdivided], [5,7])

            c0[n0_subdivided] = u5
            c2[n0_subdivided] = u7


        if is_n1_subdivided:
            u4, u6 = self.update_subdivision(n1[n1_subdivided], [4,6])

            c1[n1_subdivided] = u4
            c3[n1_subdivided] = u6

        if is_n0_subdivided or is_n1_subdivided or is_n2_subdivided or is_n3_subdivided:
            self.edge_proc_y(c2, c3, c7, c6)
            self.edge_proc_y(c0, c1, c5, c4)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def edge_proc_z(self, n0, n1, n2, n3):

        # Initialize resulting nodes to current nodes
        c0, c4 = np.copy(n0), np.copy(n0)
        c2, c6 = np.copy(n1), np.copy(n1)
        c3, c7 = np.copy(n2), np.copy(n2)
        c1, c5 = np.copy(n3), np.copy(n3)

        # Replace current nodes with their ordered children if present
        n0_subdivided, is_n0_subdivided = self.subdivided(n0)
        n1_subdivided, is_n1_subdivided = self.subdivided(n1)
        n2_subdivided, is_n2_subdivided = self.subdivided(n2)
        n3_subdivided, is_n3_subdivided = self.subdivided(n3)

        if is_n2_subdivided:
            u0, u4 = self.update_subdivision(n2[n2_subdivided], [0,4])

            c3[n2_subdivided] = u0
            c7[n2_subdivided] = u4

        if is_n3_subdivided:
            u2,u6 = self.update_subdivision(n3[n3_subdivided], [2,6])
            
            c1[n3_subdivided] = u2
            c5[n3_subdivided] = u6

        if is_n0_subdivided:
            u3, u7 = self.update_subdivision(n0[n0_subdivided], [3, 7])

            c0[n0_subdivided] = u3
            c4[n0_subdivided] = u7

        if is_n1_subdivided:
            u1, u5 = self.update_subdivision(n1[n1_subdivided], [1, 5])

            c2[n1_subdivided] = u1
            c6[n1_subdivided] = u5

        if is_n0_subdivided or is_n1_subdivided or is_n2_subdivided or is_n3_subdivided:
            self.edge_proc_z(c4, c6, c7, c5)
            self.edge_proc_z(c0, c2, c3, c1)

            self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)

    def vert_proc(self, n0, n1, n2, n3, n4, n5, n6, n7):
        
        nds = [n0, n1, n2, n3, n4, n5, n6, n7]
        
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
            
            vt = np.array([nj[inds]['centre'] for nj in nds])
            self.vertices.append(np.swapaxes(vt,0, 1))
            
            vv = np.array([nj[inds]['nPoints'] * self._density_sc[nj[inds]['depth']] for nj in nds])
            self.values.append(np.swapaxes(vv,0, 1))

            vd = np.array([nj[inds]['depth'] for nj in nds])
            self.depths.append(np.swapaxes(vd, 0, 1))

        if np.sum(~(leaf_nodes).astype(bool)) > 0:

            inds = np.where(~(leaf_nodes).astype(bool))

            # Initialize resulting nodes to current nodes
            c0, c1, c2, c3, c4, c5, c6, c7 = [np.copy(nj[inds]) for nj in nds]
            

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
                u7, = self.update_subdivision(c0[c0_subdivided], [7])
                c0[c0_subdivided] = u7
            if is_c1_subdivided:
                u6, = self.update_subdivision(c1[c1_subdivided], [6])
                c1[c1_subdivided] = u6
            if is_c2_subdivided:
                u5,  = self.update_subdivision(c2[c2_subdivided], [5])
                c2[c2_subdivided] = u5
            if is_c3_subdivided:
                u4, = self.update_subdivision(c3[c3_subdivided], [4])
                c3[c3_subdivided] = u4
            if is_c4_subdivided:
                u3, = self.update_subdivision(c4[c4_subdivided], [3])
                c4[c4_subdivided] = u3
            if is_c5_subdivided:
                u2, = self.update_subdivision(c5[c5_subdivided], [2])
                c5[c5_subdivided] = u2
            if is_c6_subdivided:
                u1, = self.update_subdivision(c6[c6_subdivided], [1])
                c6[c6_subdivided] = u1
            if is_c7_subdivided:
                u0, = self.update_subdivision(c7[c7_subdivided], [0])
                c7[c7_subdivided] = u0

            if is_c0_subdivided or is_c1_subdivided or is_c2_subdivided or is_c3_subdivided or is_c4_subdivided or is_c5_subdivided or is_c6_subdivided or is_c7_subdivided:
                self.vert_proc(c0, c1, c2, c3, c4, c5, c6, c7)


class PiecewiseDualMarchingCubes(DualMarchingCubes):
    """
    Swaps out the interpolation rountine for one which segments each edge into pieces, weighted by the vertex areas
    of each vertex
    """
    
    def interpolate_vertex(self, v0, v1, v0_value, v1_value, i=slice(None), j0=None, j1=None):
        """
        Interpolate triangle vertex along edge formed by v0->v1.

        Parameters
        ----------
        v0, v1 :
            Vertices of edge v0->v1.
        v0_value, v1_value:
            Scalar values at vertices v0, v1. Same values as for vertex_values in edge_index().

        Returns
        -------
        Interpolated vertex of a triangle.

        """
        
        depth0 = self.depths[i, j0]
        depth1 = self.depths[i, j1]
        
        d_depth = depth0 - depth1
        
        f = 8 ** d_depth
        r = 1.0 / (1 + f)
        
        #print(f[~(f==1)], r[~(f==1)])
        
        #m_value = 0.5*(v0_value + v1_value)\
        #r = depth1/(depth0 + depth1)
        m_value = (v0_value * r + v1_value * (1 - r))
        #vm = (v0*depth1[:,None] + v1*depth0[:,None])/(depth1 +depth0)[:,None]
        vm = v0 * (1 - r[:, None]) + v1 * r[:, None]
        
        #print(depth0[~(r == 0.5)], depth1[~(r == 0.5)], r[~(r == 0.5)])
        
        #print(v0_value[~(r == 0.5)], v1_value[~(r == 0.5)], m_value[~(r == 0.5)])
        
        # Interpolate along the edge v0 -> v1
        mu1 = 1. * (self.isolevel - v0_value) / (m_value - v0_value)
        p = v0 + mu1[:, None] * (vm - v0)
        
        mu2 = 1. * (self.isolevel - m_value) / (v1_value - m_value)
        p[mu2 > 0, :] = (vm + mu2[:, None] * (v1 - vm))[mu2 > 0, :]
        #print(mu1, mu2)
        
        # Are v0 and v1 the same vertex? (common in dual marching cubes)
        # If so, choose v0 as the triangle vertex position.
        idxs = (np.abs(v1_value - v0_value) < 1e-12)#self.isolevel)
        p[idxs, :] = v0[idxs, :]
        
        return p