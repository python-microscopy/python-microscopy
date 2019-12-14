def test_insert():
    import PYME.experimental._treap as treap
    import random

    h = treap.Treap()
    cost_key = [(1, 12), (3, 19), (6, 14), (5, 17), (2, 7), (7, 6), (4, 8), (9, 0), (8, 11)]
    random.shuffle(cost_key)
    for c, k in cost_key:
        h.insert(c, k)

    root_node = h._nodes[h._root_node]
    t0 = (root_node['key'] == 12)
    t1 = (h._nodes['key'][root_node['left']] == 7) & (h._nodes['key'][root_node['right']] == 19)
    t2 = (h._nodes['parent'][root_node['left']] == h._root_node) & (h._nodes['parent'][root_node['right']] == h._root_node)
    t3 = (h._nodes['key'][h._nodes['left'][root_node['left']]] == 6) & (h._nodes['key'][h._nodes['right'][root_node['left']]] == 8)
    t4 = (h._nodes['parent'][h._nodes['left'][root_node['left']]] == root_node['left']) & (h._nodes['parent'][h._nodes['right'][root_node['left']]] == root_node['left'])
    t5 = (h._nodes['key'][h._nodes['left'][root_node['right']]] == 17) & (h._nodes['key'][h._nodes['right'][root_node['right']]] == -1)
    t6 = (h._nodes['parent'][h._nodes['left'][root_node['right']]] == root_node['right'])
    t7 = (h._nodes['key'][h._nodes['left'][h._nodes['left'][root_node['right']]]] == 14) & (h._nodes['key'][h._nodes['right'][h._nodes['left'][root_node['right']]]] == -1)
    t8 = (h._nodes['parent'][h._nodes['left'][h._nodes['left'][root_node['right']]]] == h._nodes['left'][root_node['right']])
    t9 = (h._nodes['key'][h._nodes['left'][h._nodes['left'][root_node['left']]]] == 0) & (h._nodes['key'][h._nodes['right'][h._nodes['left'][root_node['left']]]] == -1)
    t10 = (h._nodes['parent'][h._nodes['left'][h._nodes['left'][root_node['left']]]] == h._nodes['left'][root_node['left']])
    t11 = (h._nodes['key'][h._nodes['left'][h._nodes['right'][root_node['left']]]] == -1) & (h._nodes['key'][h._nodes['right'][h._nodes['right'][root_node['left']]]] == 11)
    t12 = (h._nodes['parent'][h._nodes['right'][h._nodes['right'][root_node['left']]]] == h._nodes['right'][root_node['left']])

    assert(t0 & t1 & t2 & t3 & t4 & t5 & t6 & t7 & t8 & t9 & t10 & t11 & t12)

def test_delete():
    import PYME.experimental._treap as treap
    import random

    h = treap.Treap()
    cost_key = [(1, 12), (3, 19), (6, 14), (5, 17), (2, 7), (7, 6), (4, 8), (9, 0), (8, 11)]
    random.shuffle(cost_key)
    for c, k in cost_key:
        h.insert(c, k)

    h.delete(19)

    # The change occurred
    t_change = (h._nodes['key'][h._nodes['right'][h._root_node]] == 17)

    # But the left size of the tree stayed the same
    root_node = h._nodes[h._root_node]
    t0 = (root_node['key'] == 12)
    t1 = (h._nodes['key'][root_node['left']] == 7) 
    t2 = (h._nodes['parent'][root_node['left']] == h._root_node) 
    t3 = (h._nodes['key'][h._nodes['left'][root_node['left']]] == 6) & (h._nodes['key'][h._nodes['right'][root_node['left']]] == 8)
    t4 = (h._nodes['parent'][h._nodes['left'][root_node['left']]] == root_node['left']) & (h._nodes['parent'][h._nodes['right'][root_node['left']]] == root_node['left'])
    t9 = (h._nodes['key'][h._nodes['left'][h._nodes['left'][root_node['left']]]] == 0) & (h._nodes['key'][h._nodes['right'][h._nodes['left'][root_node['left']]]] == -1)
    t10 = (h._nodes['parent'][h._nodes['left'][h._nodes['left'][root_node['left']]]] == h._nodes['left'][root_node['left']])
    t11 = (h._nodes['key'][h._nodes['left'][h._nodes['right'][root_node['left']]]] == -1) & (h._nodes['key'][h._nodes['right'][h._nodes['right'][root_node['left']]]] == 11)
    t12 = (h._nodes['parent'][h._nodes['right'][h._nodes['right'][root_node['left']]]] == h._nodes['right'][root_node['left']])

    # TODO: Write a better test for delete

    assert (t_change & t0 & t1 & t2 & t3 & t4 & t9 & t10 & t11 & t12)