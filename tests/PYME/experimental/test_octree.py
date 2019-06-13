import numpy as np

def test_octree_subdiv():
    from PYME.experimental import _octree as octree
    
    
    for i in range(100):
        #error occurs randomly (1 in 8 chance)
        pts = np.random.rand(1000, 3).astype('f4')
        ot = octree.Octree([0, 1, 0, 1, 0, 1])
        
        ot.add_points(pts)
        
        if not ((ot._nodes['depth']== 1).sum() == 8):
            print(ot._nodes[ot._nodes['depth']==1])
            print(np.where(ot._nodes['depth'] == 1))
        
        assert((ot._nodes['depth']== 1).sum() == 8)

def test_octree_maxdepth_5():
    from PYME.experimental import _octree as octree

    pts = np.random.rand(100000, 3).astype('f4')
    ot = octree.Octree([0, 1, 0, 1, 0, 1], maxdepth=5)

    ot.add_points(pts)
    
    print('max depth: %d ' % max(ot._nodes['depth']))

    assert (max(ot._nodes['depth']) == 5)


def test_octree_maxdepth_8():
    from PYME.experimental import _octree as octree
    
    pts = np.random.rand(100000, 3).astype('f4')
    ot = octree.Octree([0, 1, 0, 1, 0, 1], maxdepth=8)
    
    ot.add_points(pts)
    
    print('max depth: %d ' % max(ot._nodes['depth']))
    
    assert (max(ot._nodes['depth']) == 8)
    
if __name__ == '__main__':
    import time
    t1 = time.time()
    test_octree()
    print('Created octree in %3.2fs' % (time.time() - t1))
    