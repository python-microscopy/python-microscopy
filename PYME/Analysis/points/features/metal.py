"""
Metal (apple GPU accelerated) inplementation of point features

#TODO - c/CUDA/OpenCL implementation(s)
#TODO - KDTree pre-selection for really big datasets
"""
import os
import numpy as np




class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Backend(object, metaclass=Singleton):
    def __init__(self):
        import metalcompute as mc # defer import so we fail late if mc not present
        
        self.dev = mc.Device()
        # keep the actual shader code in a separate file to get syntax highlighting etc ...
        with open(os.path.join(os.path.dirname(__file__), 'shaders', 'vect_feat_3d.metal')) as f:
            code = f.read()

        self._vect_polar_feat_3d = self.dev.kernel(code).function('vect_polar_feat_3d')

    def vector_features_3d(self, x, y, z, radial_bin_size=10, n_radial_bins=10, n_angle_bins=10):
        N = len(x)

        assert(len(y) == N)
        assert(len(z) == N)
        assert((n_angle_bins%2) ==0) #must be even
        
        n_bins_phi = int(n_angle_bins/2)
        
        buf_out = self.dev.buffer(N*n_radial_bins*n_angle_bins*n_bins_phi*4)
        h = self._vect_polar_feat_3d(N, np.array([n_radial_bins, n_angle_bins,radial_bin_size, N], 'i4'), x, y, z, x, y, z, buf_out)
        del h # block waiting for computation
        o = np.frombuffer(buf_out, 'i4')
        #o.max()
    
        return o.reshape([N, n_radial_bins, n_angle_bins, n_bins_phi])
