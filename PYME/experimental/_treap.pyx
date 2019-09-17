cimport numpy as np
import numpy as np
cimport cython

INITIAL_NODES = 1000
NODE_DTYPE = [('left', 'i4'), ('right', 'i4'), ('parent', 'i4'), ('priority', 'f4'), ('key', 'i4')]

cdef packed struct node_d:
    np.int32_t left
    np.int32_t right
    np.int32_t parent
    np.float32_t priority
    np.int32_t key

cdef class Treap:
    """
    Implement a treap (tree (min-)heap): a binary tree in which every node has
    a search key and a priority. This is essentially a 2D priority queue. We 
    have fast access to the element of minimum priority, but we can also 
    quickly find and delete nodes based on their search key.

    Another name for this data structure is, priority search tree.

    References
    ----------
       Lecture notes "Lecture 8: Treaps and Skip Lists" by Jeff Erickson
    """
    cdef public object _nodes
    cdef list _empty
    cdef public np.int32_t _root_node
    cdef node_d * _cnodes
    
    def __init__(self):
        self._nodes = np.zeros(INITIAL_NODES, NODE_DTYPE)
        self._nodes[:] = -1
        self._empty = list(np.arange(1,INITIAL_NODES)[::-1])
        self._root_node = 0

        self._set_cnodes(self._nodes)

    def _set_cnodes(self, node_d[:] nodes):
        self._cnodes = &nodes[0]

    def _search(self, np.int32_t key, np.int32_t starting_node=-2):
        """
        Return the index _curr of the node with key key.

        Parameters
        ----------
            key : int
                search key
            starting_node : int
                Index of the node at which to start the search.

        Returns
        -------
            _curr : int
                Index of node with key key.
        """
        cdef np.int32_t _curr
        cdef node_d * curr_node

        if starting_node == -2:
            starting_node = self._root_node
        _curr = starting_node
        while _curr != -1:
            curr_node = &self._cnodes[_curr]
            if key == curr_node.key:
                return _curr
            if key < curr_node.key:
                _curr = curr_node.left
            else:
                _curr = curr_node.right
        return _curr

    def _resize(self):
        old_nodes = self._nodes
        new_size = int(old_nodes.shape[0]*1.5 + 0.5)
        self._nodes = np.zeros(new_size, NODE_DTYPE)
        self._set_cnodes(self._nodes)
        self._nodes[:] = -1
        self._nodes[:old_nodes.shape[0]] = old_nodes
        self._empty = self._empty[::-1]
        self._empty.extend(list(np.arange(old_nodes.shape[0], new_size)))
        self._empty = self._empty[::-1]

    def _rotate(self, np.int32_t v):
        """
        Rotate (left or right) the node ordering of the node at index v.
        
        Parameters
        ---------
            v : int
                Index of node to rotate.
        """

        cdef np.int32_t parent, grandparent

        parent = self._cnodes[v].parent
        grandparent = self._cnodes[parent].parent

        if self._cnodes[parent].left == v:
            # Rotate right
            self._cnodes[parent].left = self._cnodes[v].right
            if (self._cnodes[v].right != -1):
                self._cnodes[self._cnodes[v].right].parent = parent
            self._cnodes[v].right = parent
        elif self._cnodes[parent].right == v:
            # Rotate left
            self._cnodes[parent].right = self._cnodes[v].left
            if self._cnodes[v].left != -1:
                self._cnodes[self._cnodes[v].left].parent = parent
            self._cnodes[v].left = parent
        else:
            print('broken')
            return

        self._cnodes[parent].parent = v
        self._cnodes[v].parent = grandparent

        if parent == self._root_node:
            # Rotating up
            self._root_node = v
        
        if (grandparent != -1):
            if self._cnodes[grandparent].right == parent:
                self._cnodes[grandparent].right = v
            elif self._cnodes[grandparent].left == parent:
                self._cnodes[grandparent].left = v

    def insert(self, priority, key, root=None):
        """
        Insert a new node.

        Parameters
        ----------
            priority : float
                Priority of this node
            key : int
                Searchable key for this node
        """
        cdef np.int32_t _curr
        cdef node_d *curr_node

        if key == -1:
            # Can't insert something that doesn't exist
            return

        if root is None:
            root = self._root_node

        if len(self._empty) == 0:
            self._resize()

        # Standard binary tree insert
        _curr = root
        
        curr_node = &self._cnodes[_curr]
        curr_key = curr_node.key
        while curr_key != -1:
            root = _curr
            if curr_key == key:
                curr_node.priority = priority
                return
            if key > curr_key:
                if curr_node.right == -1:
                    curr_node.right = self._empty.pop()
                _curr = curr_node.right
            else:
                if curr_node.left == -1:
                    curr_node.left = self._empty.pop()
                _curr = curr_node.left
            curr_node = &self._cnodes[_curr]
            curr_key = curr_node.key

        curr_node.key = key
        curr_node.priority = priority
        curr_node.parent = root
        if _curr == self._root_node:
            curr_node.parent = -1
        else:
            curr_node.parent = root

        # Rotate to maintain heap properties
        while (self._cnodes[_curr].parent != -1) and ((priority) < (self._cnodes[self._cnodes[_curr].parent].priority)):
            self._rotate(_curr)
    
    def delete(self, np.int32_t key):
        """
        Delete node with key key.

        Parameters
        ----------
            key : int
                Searchable key identifying node to delete
        """

        cdef np.int32_t _curr
        cdef node_d *curr_node

        # Find the node with key key
        _curr = self._search(key)

        if _curr == -1:
            # Can't delete something that doesn't exist, carry on
            return

        curr_node = &self._cnodes[_curr]

        # Make sure node with key key is a leaf
        while curr_node.left != -1 or curr_node.right != -1:
            # Choose the minimumum priority node to rotate up (rotate _curr down)
            if curr_node.right == -1:
                _to_rotate = curr_node.left
            elif curr_node.left == -1:
                _to_rotate = curr_node.right
            elif ((self._cnodes[curr_node.left].priority) > (self._cnodes[curr_node.right].priority)):
                _to_rotate = curr_node.right
            else:
                _to_rotate = curr_node.left
            self._rotate(_to_rotate)
        
        # Now delete the node
        if self._cnodes[curr_node.parent].left == _curr:
            self._cnodes[curr_node.parent].left = -1
        else:
            self._cnodes[curr_node.parent].right = -1
        self._cnodes[_curr].left = -1
        self._cnodes[_curr].right = -1
        self._cnodes[_curr].parent = -1
        self._cnodes[_curr].priority = -1
        self._cnodes[_curr].key = -1

        # Mark the free space as available
        self._empty.append(_curr)

    def peek(self):
        """
        Look at the element with smallest priority in the treap.

        Returns
        ----------
            priority : float
                Priority of smallest node
            key : int
                Searchable key of smllest node
        """
        node = self._nodes[self._root_node]
        return node['priority'], node['key']

    def pop(self):
        """
        Get the element of smallest priority in the treap and remove from 
        the treap.

        Returns
        ----------
            priority : float
                Priority of smallest node
            key : int
                Searchable key of smllest node
        """
        node = self._nodes[self._root_node]
        priority, key = node['priority'], node['key']
        self.delete(node['key'])
        return priority, key