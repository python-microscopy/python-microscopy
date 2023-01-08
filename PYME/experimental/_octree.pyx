cimport numpy as np
import numpy as np
cimport cython

#size to initialize the storage to
INITIAL_NODES = 1000
NODE_DTYPE = [('depth', 'i4'), ('n_children', 'i4'), ('children', '8i4'),('parent', 'i4'), ('nPoints', 'i4'), ('centre', '3f4'), ('centroid', '3f4'),('point_idx', 'i4')]

#flat dtype to work around a cython memory view bug with array dtypes
NODE_DTYPE2 = [('depth', 'i4'),
               ('n_children', 'i4'),
               ('child0', 'i4'),
               ('child1', 'i4'),
               ('child2', 'i4'),
               ('child3', 'i4'),
               ('child4', 'i4'),
               ('child5', 'i4'),
               ('child6', 'i4'),
               ('child7', 'i4'),
               ('parent', 'i4'),
               ('nPoints', 'i4'),
               ('centre_x', 'f4'),
               ('centre_y', 'f4'),
               ('centre_z', 'f4'),
               ('centroid_x', 'f4'),
               ('centroid_y', 'f4'),
               ('centroid_z', 'f4'),
               ('point_idx', 'i4')]

#the struct that the above dtype maps to in c
cdef packed struct node_d:
    np.int32_t depth
    np.int32_t n_children
    np.int32_t child0
    np.int32_t child1
    np.int32_t child2
    np.int32_t child3
    np.int32_t child4
    np.int32_t child5
    np.int32_t child6
    np.int32_t child7
    np.int32_t parent
    np.int32_t nPoints
    np.float32_t centre_x
    np.float32_t centre_y
    np.float32_t centre_z
    np.float32_t centroid_x
    np.float32_t centroid_y
    np.float32_t centroid_z
    np.int32_t point_idx
    
    
cdef float[8] _octant_sign_x
cdef float[8] _octant_sign_y
cdef float[8] _octant_sign_z
cdef int n

for n in range(8):
    _octant_sign_x[n] = 2*(n&1) - 1
    _octant_sign_y[n] = (n&2) -1
    _octant_sign_z[n] = (n&4)/2.0 -1


cdef class Octree:
    '''Implement an octree as a doubly linked tree inside a pre-allocate numpy array, with resizing on growth
    (as done for vector types in C++ std lib)
    
    This is designed to store points, and current does not store any additional data with the point data
     - this would be reasonably easy to add by modifying the dtype - an index back to the original data source would be
     the first candidate
     
     Features:
     ---------
     - parent and child relations are stored as offets (indices) into the flat storage array
     - the 0th entry is the root of the tree
     - The leaf and non-leaf nodes use the same data structure with the leaf nodes being distinguished as follows:
        - leaf nodes have nPoints = 1
        - all child indices in leaf nodes should be 0
    - there is one localization / point per leaf
    - the 'centroid' entry is meaningful for leaves, in which case it indicates the position of the corresponding
      localization / point. It is currently not meaningful for non-leaf nodes, where it is purely  an artifact of how the tree
      was created (all nodes start as a leaf and are then subdivided when new data would fall into them).
    - there is not much implemented for doing stuff with the points once they are in the tree although the flat tree
      data is accessible (as a numpy array) through the _nodes property
      
    the _nodes property:
    --------------------
    - this gives access to flattened node data
    - it can easily filtered using standard numpy commands - e.g. _nodes[_nodes['depth'] == 5]
    - it should be fairly easy to plot/visualize the tree based on this flattened data
      
    The code is reasonably optimized with creation of a 1M entry octree taking 2-3s
    
    '''
    cdef int _next_node, _resize_limit, _maxdepth, _samples_per_node
    cdef np.float32_t _xwidth, _ywidth, _zwidth
    cdef np.float32_t[6] _bounds
    cdef np.float32_t[3] _size
    cdef public object _nodes
    cdef object _flat_nodes
    cdef node_d * _cnodes
    cdef object _octant_offsets
    cdef object _octant_sign
    
    cdef np.float32_t[50] _scale

    cdef public object mdh # placeholder to allow metadata injection
    cdef public object points # placeholder to allow points injection
    
    def __init__(self, bounds, maxdepth=12, samples_per_node=1):
        cdef int n
        
        bounds = np.array(bounds, 'f4')
        self._bounds = bounds
        
        self._xwidth = bounds[1] - bounds[0]
        self._ywidth = bounds[3] - bounds[2]
        self._zwidth = bounds[5] - bounds[4]
        
        self._size= np.array([self._xwidth, self._ywidth, self._zwidth], 'f4')
        
        self._nodes = np.zeros(INITIAL_NODES, NODE_DTYPE)
        
        self._nodes[0]['centre'][:] = [(bounds[1] + bounds[0])/2.0, (bounds[3] + bounds[2])/2.0, (bounds[5] + bounds[4])/2.0]
        self._flat_nodes = self._nodes.view(NODE_DTYPE2)
        self._set_cnodes(self._flat_nodes)
        
        self._maxdepth = maxdepth
        self._samples_per_node = samples_per_node
        
        self._next_node = 1
        self._resize_limit = INITIAL_NODES - 1
        
        self._octant_offsets = np.array([1, 2, 4])
        
        self._octant_sign = np.array([[2*(n&1) - 1, (n&2) -1, (n&4)/2 -1] for n in range(8)])

        self.mdh = None
          
        #precalculate scaling factors for locating the centre of the next box down.
        for n in range(50):
            self._scale[n] = 1.0/(2.0**(n + 1))
            
        
    def truncate_at_n_points(self, n_points=5):
        out = Octree(self._bounds, self._maxdepth, samples_per_node=n_points)
        
        # n2 = np.copy(self._nodes[self._nodes[self._nodes['parent']]['nPoints'] >= n_points])
        # n2[n2['nPoints'] <= n_points]['children'] = 0
        #
        # out._nodes = np.zeros(len(n2) + 1, NODE_DTYPE)
        # out._nodes [:-1] = n2
        # out._next_node = len(n2)
        
        out._nodes = np.copy(self.nodes)
        out._nodes['children'][out._nodes['nPoints'] <= n_points] = 0
        
        return out
        
        
    def _set_cnodes(self, node_d[:] nodes):
        self._cnodes = &nodes[0]
        
    def _search(self, pos):
        node_idx, child_idx, subdivide =  self.search(pos[0], pos[1], pos[2])
        return node_idx, self._nodes[node_idx], child_idx, subdivide
    
    @property
    def nodes(self):
        return self._nodes[:self._next_node]

    
    cpdef search_pts(self, np.float32_t [:, :] pts):
        """
        return (approximate) nearest vertex indices for a specific set of test points
        """

        cdef int i
        cdef int [:] _idxs

        idxs = np.zeros(pts.shape[0], 'i')
        _idxs = idxs

        for i in range(pts.shape[0]):
            node_idx, _, _ = self.search(pts[i, 0], pts[i, 1], pts[i,2])

            idxs[i] = self._cnodes[node_idx].point_idx

        return idxs
    
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    cpdef search(self, np.float32_t x, np.float32_t y, np.float32_t z):
        cdef int node_idx, child_idx
        #cdef node_d * p_nodes
        cdef node_d node
        cdef np.int32_t *children
        #p_nodes = <node_d*>&nodes[0]
        node_idx = 0
        node = self._cnodes[node_idx]
        
        
        #print 'getting children'
        child_idx = ((x > node.centre_x) + 2*(y > node.centre_y)  + 4*(z > node.centre_z))
        while node.nPoints >= self._samples_per_node:
            children = &node.child0
            new_idx = children[child_idx]
            if new_idx == 0:
                #node subdivided but child is not yet allocated, no need to subdivide
                return node_idx,  child_idx, False
            else:
                node_idx = new_idx
            
            node = self._cnodes[node_idx]
            child_idx = ((x > node.centre_x) + 2*(y > node.centre_y)  + 4*(z > node.centre_z))
            #visited.append(node_idx)
            
        if node_idx == 0:
            #node subdivided but child is not yet allocated, no need to subdivide
            return node_idx, child_idx, False
            
        #node has not been subdivided
        return node_idx, child_idx, True
         
    
    def _resize_nodes(self):
        old_nodes = self._nodes
        # Adjust by 1.5x every time we grow.
        new_size = int(old_nodes.shape[0]*1.5 + 0.5)
        
        print('Resizing node store - new size: %d' % new_size)
        #allocate new memory
        self._nodes = np.zeros(new_size, NODE_DTYPE)
        self._flat_nodes = self._nodes.view(NODE_DTYPE2)
        self._set_cnodes(self._flat_nodes)
        
        #copy data over TODO - use a direct call to memcpy instead?
        self._nodes[:self._next_node] = old_nodes[:self._next_node]
        self._resize_limit = new_size - 1

    def _add_node(self, pos, parent_idx, child_idx):
        new_idx =  self.__add_node(pos[0], pos[1], pos[2], parent_idx, child_idx)
        
        return new_idx, self._nodes[new_idx]
    
    
    def box_size(self, depth):
        scale = 1.0/(2**depth)
        
        deltax = self._xwidth*scale
        deltay = self._ywidth*scale
        deltaz = self._zwidth*scale
        
        return deltax, deltay, deltaz
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    def __add_node(self, np.float32_t x, np.float32_t y, np.float32_t z, int parent_idx,
                  int child_idx, int point_idx):
        cdef int new_idx
        cdef float scale
        cdef np.int32_t *children
        cdef node_d *new_node, *parent
        
        
        if self._next_node >= self._resize_limit:
            self._resize_nodes()
    
        new_idx = self._next_node
        self._next_node += 1
        
        #cdef node_d parent, new_node
        parent = &self._cnodes[parent_idx]
        
        scale = 1.0/(2.0**(self._cnodes[parent_idx].depth + 2))
        
        deltax = self._xwidth*scale
        deltay = self._ywidth*scale
        deltaz = self._zwidth*scale
        
        new_node = &self._cnodes[new_idx]
        
        new_node.nPoints = 1
        new_node.centre_x = parent.centre_x + _octant_sign_x[child_idx]*deltax
        new_node.centre_y = parent.centre_y + _octant_sign_y[child_idx]*deltay
        new_node.centre_z = parent.centre_z + _octant_sign_z[child_idx]*deltaz
    
        new_node.centroid_x = x
        new_node.centroid_y = y
        new_node.centroid_z = z
        
        new_node.depth = parent.depth + 1
        new_node.parent = parent_idx

        new_node.point_idx = point_idx
        
        children = &parent.child0
        children[child_idx] = new_idx
        
        return new_idx

    cpdef add_point(self, np.float32_t[:] pos, int pt_idx=0):
        cdef float x,y,z
        cdef int node_idx, child_idx
        cdef bint subdivide
        
        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        #find the node we need to subdivide
        node_idx, child_idx, subdivide = self.search(x, y, z)
        
        c_node = self._cnodes[node_idx]
        
        if c_node.depth < self._maxdepth:
            if not subdivide:
                #add a new node
                #print 'adding without subdivision'
                
                self.__add_node(x, y, z, node_idx, child_idx, pt_idx)
            else:
                #print 'adding with subdivision: ',  node_idx, child_idx
                #we need to move the current node data into a child
                
                #child to put the data currently held by the node into
                node_child_idx = ((x > c_node.centre_x) + 2*(y > c_node.centre_y)  + 4*(z > c_node.centre_z))
                #make a new node with the current nodes data
                new_idx = self.__add_node(c_node.centroid_x, c_node.centroid_y, c_node.centroid_z, node_idx, node_child_idx, c_node.point_idx)
                
                #continue subdividing until original and new data do not want to go in the same child
                while (node_child_idx == child_idx) and (c_node.depth < (self._maxdepth - 1)):
                    node_idx = new_idx
                    c_node = self._cnodes[node_idx]
                    
                    #child to put the new data in
                    child_idx = ((x > c_node.centre_x) + 2*(y > c_node.centre_y)  + 4*(z > c_node.centre_z))
                    
                    #child to put the data currently held by node in
                    node_child_idx = ((c_node.centroid_x > c_node.centre_x) +  2*(c_node.centroid_y > c_node.centre_y) +
                                      4*(c_node.centroid_z > c_node.centre_z))
                    
                    new_idx = self.__add_node(c_node.centroid_x, c_node.centroid_y, c_node.centroid_z, node_idx, node_child_idx, c_node.point_idx)
                
                if (c_node.depth < self._maxdepth):
                    self.__add_node(x, y, z, node_idx, child_idx, pt_idx)

        self._cnodes[node_idx].nPoints += 1
        while node_idx > 0:
            #print node_idx
            node_idx = self._cnodes[node_idx].parent
            #print node_idx, self._nodes[node_idx]['depth']
            self._cnodes[node_idx].nPoints += 1
            #TODO - update centroids
            
    def add_points(self, np.float32_t[:,:] pts):
        cdef int i

        # Double check for oversmoothing
        if len(pts) < self._samples_per_node:
            self._samples_per_node = len(pts)

        for i in range(len(pts)):
            self.add_point(pts[i], i)
            
    def update_n_children(self):
        self._nodes['n_children'] = np.sum(self._nodes['children'], axis=1)
        
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    def _fix_empty_nodes(self, node_d[:] nj, node_d[:] parent, int j):
        cdef int i
        cdef float scale, deltax, deltay, deltaz
        
        for i in range(nj.shape[0]):
            if nj[i].depth == 0:
                nj[i].depth = parent[i].depth + 1
                scale = self._scale[nj[i].depth]
        
                deltax = self._xwidth*scale
                deltay = self._ywidth*scale
                deltaz = self._zwidth*scale
        
                nj[i].centre_x = parent[i].centre_x + _octant_sign_x[j]*deltax
                nj[i].centre_y = parent[i].centre_y + _octant_sign_y[j]*deltay
                nj[i].centre_z = parent[i].centre_z + _octant_sign_z[j]*deltaz
                
                nj[i].child0 = 0
                nj[i].child1 = 0
                nj[i].child2 = 0
                nj[i].child3 = 0
                nj[i].child4 = 0
                nj[i].child5 = 0
                nj[i].child6 = 0
                nj[i].child7 = 0
                nj[i].n_children = 0
                nj[i].nPoints = 0
                
    def fix_empty_nodes(self, nj, parent, int j):
        self._fix_empty_nodes(nj.view(NODE_DTYPE2), parent.view(NODE_DTYPE2), j)
        
        
                
                
            
            
        
            
            
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
def _has_children(node_d[:] nodes):
    cdef bint any_have_children, _node_children
    cdef int i, j, N
    cdef node_d * _cnodes
    cdef np.int32_t *children
    #cdef bool * _subdiv
    cdef np.uint8_t[:] subdiv
    
    N = len(nodes)
    _cnodes = &nodes[0]
    subdiv = np.zeros(N, 'uint8')
    #_subdiv = &subdiv[0]
    
    any_have_children = False
    
    for i in range(N):
        children = &_cnodes[i].child0
        _node_children = False
        j = 0
        
        while not _node_children and j < 8:
            if children[j] > 0:
                _node_children = True
                any_have_children = True
                
            j+=1
                
        subdiv[i] = _node_children
        
    return subdiv, any_have_children
            
        
        
def has_children(nodes):
    return _has_children(nodes.view(NODE_DTYPE2))