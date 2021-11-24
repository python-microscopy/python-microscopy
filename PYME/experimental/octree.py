import numpy as np
from . _octree import Octree

INITIAL_NODES = 1000
NODE_DTYPE = [('depth', 'i4'), ('children', '8i4'),('parent', 'i4'), ('nPoints', 'i4'), ('centre', '3f4'), ('centroid', '3f4')]
NODE_DTYPE2 = [('depth', 'i4'),
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
               ('centroid_z', 'f4')]

class PyOctree(object):
    '''Implement octree as a linked list inside a numpy array, with resizing (as done for vector types in C++ std lib)'''
    def __init__(self, bounds):
        bounds = np.array(bounds, 'f4')
        self._bounds = bounds
        
        self._xwidth = bounds[1] - bounds[0]
        self._ywidth = bounds[3] - bounds[2]
        self._zwidth = bounds[5] - bounds[4]
        
        self._size= np.array([self._xwidth, self._ywidth, self._zwidth], 'f4')
        
        self._nodes = np.zeros(INITIAL_NODES, NODE_DTYPE)
        
        self._nodes[0]['centre'][:] = [(bounds[1] + bounds[0])/2.0, (bounds[3] + bounds[2])/2.0, (bounds[5] + bounds[4])/2.0]
        
        self._flat_nodes = self._nodes.view(NODE_DTYPE2)
        
        self._next_node = 1
        self._resize_limit = INITIAL_NODES - 1
        
        self._octant_offsets = np.array([1, 2, 4])
        
        self._octant_sign = np.array([[2*(n&1) - 1, (n&2) -1, (n&4)/2 -1] for n in range(8)])
        
    def __search(self, pos):
        node_idx, child_idx, subdivide =  _oct_helpers._search(self._flat_nodes, pos)
        return node_idx, self._nodes[node_idx], child_idx, subdivide
    
    def _search(self, pos):
        node_idx = 0
        node = self._nodes[node_idx]
        #visited = [node_idx,]
        child_idx = ((pos > node['centre'])*self._octant_offsets).sum()
        while node['nPoints'] > 1:
            new_idx = node['children'][child_idx]
            if new_idx == 0:
                #node subdivided but child is not yet allocated, no need to subdivide
                return node_idx, node, child_idx, False
            else:
                node_idx = new_idx
            
            node = self._nodes[node_idx]
            child_idx = ((pos > node['centre']) * self._octant_offsets).sum()
            #visited.append(node_idx)
            
        if node_idx == 0:
            #node subdivided but child is not yet allocated, no need to subdivide
            return node_idx, node, child_idx, False
            
        #node has not been subdivided
        return node_idx, node, child_idx, True
                 
        
         
    
    def _resize_nodes(self):
        old_nodes = self._nodes
        new_size = old_nodes.shape[0]*2 #double whenever we grow (TODO - what is the best growth factor)
        
        #allocate new memory
        self._nodes = np.zeros(new_size, NODE_DTYPE)
        self._flat_nodes = self._nodes.view(NODE_DTYPE2)
        
        #copy data over TODO - use a direct call to memcpy instead?
        self._nodes[:self._next_node] = old_nodes[:self._next_node]
        self._resize_limit = new_size - 1

    def __add_node(self, pos, parent_idx, child_idx):
        if self._next_node >= self._resize_limit:
            self._resize_nodes()
    
        new_idx = self._next_node
        self._next_node += 1
        
        new_idx =  _oct_helpers._add_node(self._flat_nodes, new_idx, pos, parent_idx, child_idx, self._size)
        
        return new_idx, self._nodes[new_idx]
    
    def _add_node(self, pos, parent_idx, child_idx):
        if self._next_node >= self._resize_limit:
            self._resize_nodes()
            
        parent = self._nodes[parent_idx]
        new_idx = self._next_node
        self._next_node += 1
        
        delta = self._size/(2**(parent['depth'] + 2))
        
        new_node = self._nodes[new_idx]
        
        new_node['nPoints'] = 1
        new_node['centre'] = parent['centre'] + self._octant_sign[child_idx]*delta
        new_node['centroid'] = pos
        new_node['depth'] = parent['depth'] + 1
        new_node['parent'] = parent_idx
        
        if not parent['children'][child_idx] == 0:
            raise RuntimeError('Child already allocated: parent_idx=%d, child_idx=%d, pos = %s' % (parent_idx, child_idx, pos))
        parent['children'][child_idx] = new_idx
        
        return new_idx, new_node

    def add_point(self, pos):
        #find the node we need to subdivide
        node_idx, node, child_idx, subdivide = self._search(pos)
    
        if not subdivide:
            #add a new node
            #print 'adding without subdivision'
            self._add_node(pos, node_idx, child_idx)
        else:
            #print 'adding with subdivision: ',  node_idx, child_idx
            #we need to move the current node data into a child
            node_child_idx = ((node['centroid'] > node['centre']) * self._octant_offsets).sum()
            new_idx, new_node = self._add_node(node['centroid'], node_idx, node_child_idx)
            while node_child_idx == child_idx:
                node = new_node
                node_idx = new_idx
                #visited.append(new_idx)
                child_idx = ((pos > node['centre']) * self._octant_offsets).sum()
            
                node_child_idx = ((node['centroid'] > node['centre']) * self._octant_offsets).sum()
                new_idx, new_node = self._add_node(node['centroid'], node_idx, node_child_idx)
            
            self._add_node(pos, node_idx, child_idx)

        self._nodes[node_idx]['nPoints'] += 1
        while node_idx > 0:
            #print node_idx
            node_idx = self._nodes[node_idx]['parent']
            #print node_idx, self._nodes[node_idx]['depth']
            self._nodes[node_idx]['nPoints'] += 1
            #TODO - update centroids
            
    def add_points(self, pts):
        for pt in pts:
            self.add_point(pt)


def gen_octree_from_points(table, min_pixel_size=5, max_depth=20, samples_per_node=1):
    pts = np.vstack([table['x'], table['y'], table['z']]).T.astype('f4')
    
    r_min = pts.min(axis=0) - 250
    r_max = pts.max(axis=0) + 250
    
    print(f'rmin:{r_min}, r_max:{r_max}')
    
    bbox_size = (r_max - r_min).max()
    
    bb_max = r_min + 1.1* bbox_size
    
    r_min = r_min - 0.1*bbox_size
    
    max_depth = min(max_depth, np.log2(bbox_size / min_pixel_size) + 1)
    
    ot = Octree([r_min[0], bb_max[0], r_min[1], bb_max[1], r_min[2], bb_max[2]], maxdepth=max_depth, samples_per_node=samples_per_node)
    ot.add_points(pts)
    
    return ot
        
        
            
    
        