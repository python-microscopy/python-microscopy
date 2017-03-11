import numpy as np

measurement_dtype = [('count', '<i4'),
                     ('x', '<f4'),('y', '<f4'),('z', '<f4'),
                     ('gyrationRadius', '<f4'),
                     ('axis0', '<3f4'),('axis1', '<3f4'),('axis2', '<3f4'),
                     ('sigma0', '<f4'), ('sigma1', '<f4'), ('sigma2', '<f4'),
                     ('theta', '<f4'), ('phi', '<f4')]

def measure_3d(x, y, z, output=None):
    if output is None:
        output = np.zeros(1, measurement_dtype)
    
    #count
    N = len(x)
    output['count'] = N
    
    #centroid
    xc, yc, zc = x.mean(), y.mean(), z.mean()
    
    output['x'] = xc
    output['y'] = yc
    output['z'] = zc
    
    #find mean-subtracted points
    x_, y_, z_ = x - xc, y - yc, z - zc
    
    #radius of gyration
    output['gyrationRadius'] = np.sqrt(np.mean(x_*x_ + y_*y_ + z_*z_))
    
    #principle axes
    u, s, v = np.linalg.svd(np.vstack([x_, y_, z_]).T)
    
    for i in range(3):
        output['axis%d' % i] = v[i]
        #std. deviation along axes
        output['sigma%d' % i] = s[i]/np.sqrt(N-1)
    
    pa = v[0]
    #angle of principle axis
    output['theta'] = np.arctan(pa[0]/pa[1])
    output['phi'] = np.arcsin(pa[2])
    
    #TODO - compactness based on pairwise distances?
    
    return output
    