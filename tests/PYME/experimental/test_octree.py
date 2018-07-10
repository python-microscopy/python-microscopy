import numpy as np

def test_octree():
    from PYME.experimental import _octree as octree
    
    pts = np.random.rand(100000, 3).astype('f4')
    ot = octree.Octree([0, 1, 0, 1, 0, 1])
    
    ot.add_points(pts)
    
    assert((ot._nodes['depth']== 1).sum()) == 8
    
    
    
if __name__ == '__main__':
    import time
    t1 = time.time()
    test_octree()
    print('Created octree in %3.2fs' % (time.time() - t1))
    