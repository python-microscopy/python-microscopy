import numpy as np

INITIAL_NODES = 1000
NODE_DTYPE = [('left', 'i4'), ('right', 'i4'), ('parent', 'i4'), ('priority', 'f4'), ('key', 'i4')]

class PyTreap(object):
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
    def __init__(self):
        self._nodes = np.zeros(INITIAL_NODES, NODE_DTYPE)
        self._nodes[:] = -1
        self._empty = list(np.arange(1,INITIAL_NODES)[::-1])
        self._root_node = 0

    def _search(self, key, starting_node=None):
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
        if starting_node is None:
            starting_node = self._root_node
        _curr = starting_node
        while _curr != -1:
            curr_node = self._nodes[_curr]
            if key == curr_node['key']:
                return _curr
            if key < curr_node['key']:
                _curr = curr_node['left']
            else:
                _curr = curr_node['right']
        return _curr

    def _resize(self):
        old_nodes = self._nodes
        new_size = int(old_nodes.shape[0]*1.5 + 0.5)
        self._nodes = np.zeros(new_size, NODE_DTYPE)
        self._nodes[:] = -1
        self._nodes[:old_nodes.shape[0]] = old_nodes
        self._empty = self._empty[::-1]
        self._empty.extend(list(np.arange(old_nodes.shape[0], new_size)))
        self._empty = self._empty[::-1]

    def _rotate(self, v):
        """
        Rotate (left or right) the node ordering of the node at index v.
        
        Parameters
        ---------
            v : int
                Index of node to rotate.
        """

        try:
            parent = self._nodes['parent'][v]
            grandparent = self._nodes['parent'][parent]
        except(IndexError):
            raise IndexError('Node {} has no parent.'.format(v))

        if self._nodes['left'][parent] == v:
            # Rotate right
            self._nodes['left'][parent] = self._nodes['right'][v]
            self._nodes['parent'][self._nodes['right'][v]] = parent
            self._nodes['right'][v] = parent
        elif self._nodes['right'][parent] == v:
            # Rotate left
            self._nodes['right'][parent] = self._nodes['left'][v]
            self._nodes['parent'][self._nodes['left'][v]] = parent
            self._nodes['left'][v] = parent
        else:
            print('broken')
            return

        self._nodes['parent'][parent] = v
        self._nodes['parent'][v] = grandparent

        if parent == self._root_node:
            # print(grandparent, parent, v)
            # print(self._nodes[grandparent])
            # print(self._nodes[parent])
            # print(self._nodes[v])
            # Rotating up
            self._root_node = v
        
        if (grandparent != -1):
            if self._nodes['right'][grandparent] == parent:
                self._nodes['right'][grandparent] = v
            elif self._nodes['left'][grandparent] == parent:
                self._nodes['left'][grandparent] = v

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
        if root is None:
            root = self._root_node

        if len(self._empty) == 0:
            self._resize()

        # Standard binary tree insert
        _curr = root
        curr_node = self._nodes[_curr]
        curr_key = curr_node['key']
        while curr_key != -1:
            root = _curr
            if curr_key == key:
                curr_node['priority'] = priority
                return
            if key > curr_key:
                if curr_node['right'] == -1:
                    curr_node['right'] = self._empty.pop()
                _curr = curr_node['right']
                curr_node = self._nodes[_curr]
            else:
                if curr_node['left'] == -1:
                    curr_node['left'] = self._empty.pop()
                _curr = curr_node['left']
                curr_node = self._nodes[_curr]
            curr_key = curr_node['key']

        curr_node['key'] = key
        curr_node['priority'] = priority
        curr_node['parent'] = root
        if _curr == self._root_node:
            curr_node['parent'] = -1
        else:
            curr_node['parent'] = root

        # Rotate to maintain heap properties
        while (self._nodes['parent'][_curr] != -1) and ((priority) < (self._nodes['priority'][self._nodes['parent'][_curr]])):
            self._rotate(_curr)
    
    def delete(self, key):
        """
        Delete node with key key.

        Parameters
        ----------
            key : int
                Searchable key identifying node to delete
        """

        # Find the node with key key
        _curr = self._search(key)

        if _curr == -1:
            # Can't delete something that doesn't exist, carry on
            return

        curr_node = self._nodes[_curr]

        # Make sure node with key key is a leaf
        while curr_node['left'] != -1 or curr_node['right'] != -1:
            # Choose the minimumum priority node to rotate up (rotate _curr down)
            if curr_node['right'] == -1:
                _to_rotate = curr_node['left']
            elif curr_node['left'] == -1:
                _to_rotate = curr_node['right']
            elif ((self._nodes['priority'][curr_node['left']]) > (self._nodes['priority'][curr_node['right']])):
                _to_rotate = curr_node['right']
            else:
                _to_rotate = curr_node['left']
            self._rotate(_to_rotate)
        
        # Now delete the node
        if self._nodes['left'][curr_node['parent']] == _curr:
            self._nodes['left'][curr_node['parent']] = -1
        else:
            self._nodes['right'][curr_node['parent']] = -1
        self._nodes[_curr] = -1

        # Mark the free space as available
        self._empty.append(_curr)

    def _split(self, key):
        """
        Split the treap at the node with key key.

        Parameters
        ----------
            key : int
                Searchable key identifying node to split on
        """

        # 1. Insert a new node with key key and priority -infinity. This will become the root node.
        # 2. Delete the root node. The two resulting subtrees are our desired treaps.

        raise NotImplementedError('What luck! You get to write this function.')

    def _merge(self, treap):
        """
        Merge this and another treap.

        Parameters
        ----------
            treap : Treap
                Treap object to merge with self
        """
        
        # 1. Create a dummy root whose left treap is this treap and right treap is the treap we pass in.
        # 2. Rotate the dummy root to a leaf, remove it. The remaining treap is what we want.

        raise NotImplementedError('What luck! You get to write this function.')

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