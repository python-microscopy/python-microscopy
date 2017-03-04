import numpy as np

def measure_3d(x, y, z):
    output = {}
    
    #count
    N = len(x)
    output['count'] = N
    
    #centroid
    xc, yc, zc = x.mean(), y.mean(), z.mean()
    
    output.update(x = xc, y = yc, z = zc)
    
    #find mean-subtracted points
    x_, y_, z_ = x - xc, y - yc, z - zc
    
    #radius of gyration
    output['gyrationRadius'] = np.sqrt(np.mean(x_*x_ + y_*y_ + z_*z_))
    
    #principle axes
    u, s, v = np.linalg.svd(np.vstack([x_, y_, z_]).T)
    
    #axes
    output.update({'axis%d' % i : v[i] for i in range(3)})
    #std. deviation along axes
    output.update({'sigma%d' % i : s[i]/np.sqrt(N-1) for i in range(3)})
    
    pa = v[0]
    #angle of principle axis
    output['theta'] = np.arctan(pa[0]/pa[1])
    output['phi'] = np.arcsin(pa[2])
    
    #TODO - compactness based on pairwise distances?
    
    return output
    