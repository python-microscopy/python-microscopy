import numpy as np

def test_single_point_2d():
    from PYME.Analysis.points import DistHist

    min_radius = 1
    max_radius = 100
    step = 1

    radii = np.arange(min_radius,max_radius,step)-0.5
    n_points = np.round(np.random.rand(len(radii))*100).astype('int')
    points = np.array([0,0])
    for _i, r in enumerate(radii):
        x, y = np.random.randn(2,n_points[_i])
        l = np.sqrt(x**2 + y**2)
        points = np.vstack([points,np.vstack([x/l, y/l]).T*r])
    
    hist = DistHist.distanceHistogram(points[0,0], points[0,1], points[:,0], points[:,1], max_radius-min_radius, step)

    np.testing.assert_array_equal(hist,n_points)


def test_single_point_3d():
    from PYME.Analysis.points import DistHist

    min_radius = 1
    max_radius = 100
    step = 1

    radii = np.arange(min_radius,max_radius,step)-0.5
    n_points = np.round(np.random.rand(len(radii))*100).astype('int')
    points = np.array([0,0,0])
    for _i, r in enumerate(radii):
        x, y, z = np.random.randn(3,n_points[_i])
        l = np.sqrt(x**2 + y**2 + z**2)
        points = np.vstack([points,np.vstack([x/l, y/l, z/l]).T*r])
    
    hist = DistHist.distanceHistogram3D(points[0,0], points[0,1], points[0,2], points[:,0], points[:,1], points[:,2], max_radius-min_radius, step)

    np.testing.assert_array_equal(hist,n_points)

