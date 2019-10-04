from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
from PYME.IO import tabular



@register_module('Octree')
class Octree(ModuleBase):
    input_localizations = Input('input')
    output_octree = Output('output')
    
    minimum_pixel_size = Float(5)
    max_depth = Int(20)
    samples_per_node = Int(1)
    
    #bounds_mode = Enum(['Auto', 'Manual'])
    #manual_bounds = ListFloat([0,0,0, 5e3,5e3, 5e3])
    
    def execute(self, namespace):
        from PYME.experimental.octree import gen_octree_from_points
        inp = namespace[self.input_localizations]

        ot = gen_octree_from_points(inp, min_pixel_size=self.minimum_pixel_size, max_depth=self.max_depth, samples_per_node=self.samples_per_node)
        
        namespace[self.output_octree] = ot
        
        
@register_module('LocalPointDensity')
class LocalPointDensity(ModuleBase):
    """
    Estimate the local density around a localization by fitting a scaling function to the number of
    Neigbours vs distance. The expected scaling function for a uniform density is used ($N \propto r^2$
    for 2D, $N\propto r^3$ for 3D.
    
    TODO - find the correct scaling factors (probably involving pi) to convert this to $locs/um^{N_{dim}}$
    
    """
    input = Input('input')
    output = Output('output')
    
    n_nearest_neighbours = Int(10)
    three_d = Bool(True)
    
    def _dn2d(self, pt, kdt, N, n):
        """
        Find the local density by fitting a quadratic to the distance between a point and it's neighbours
        
        TODO = scaling factors?
        """
        d, _ = kdt.query(pt, N)
        return float(np.linalg.lstsq(np.atleast_2d(d**2).T, n, rcond=None)[0])

    def _dn3d(self, pt, kdt, N, n):
        """
        Find the local density by fitting a cubic to the distance between a point and it's neighbours

        TODO = scaling factors?
        """
        d, _ = kdt.query(pt, N)
        return float(np.linalg.lstsq(np.atleast_2d(d**3).T, n, rcond=None)[0])
    
    def execute(self, namespace):
        from scipy.spatial import cKDTree
        from PYME.IO import tabular
        
        inp = namespace[self.input]
        
        N = int(self.n_nearest_neighbours)
        n_ = np.arange(N)
        
        if self.three_d:
            pts = np.vstack([inp['x'], inp['y'], inp['z']]).T
            kdt = cKDTree(pts)
        
            d_ = np.array([self._dn3d(p, kdt, N, n_) for p in pts])
        else:
            pts = np.vstack([inp['x'], inp['y']]).T
            kdt = cKDTree(pts)
    
            d_ = np.array([self._dn2d(p, kdt, N, n_) for p in pts])
            
        t = tabular.mappingFilter(inp)
        t.addColumn('dn', d_)
        
        try:
            t.mdh = inp.mdh
        except AttributeError:
            pass
        
        namespace[self.output] = t
        
        
from PYME.experimental.triangle_mesh import TrianglesBase
class Tesselation(tabular.TabularBase, TrianglesBase):
    """
    A wrapper class which encapsulates both a tesselation and the underlying
    point data source
    
    FIXME - move somewhere more sensible
    """
    def __init__(self, point_data, three_d = True):
        from scipy.spatial import Delaunay
        self._3d = three_d
        
        self._data = point_data
        self.vertices = np.vstack([self._data['x'], self._data['y'], self._data['z']]).T
        
        if self._3d:
            self.T = Delaunay(self.vertices)
        else:
            self.vertices[:,2]= 0 #set z to 0
            self.T = Delaunay(self.vertices[:,:2])
            
    def keys(self):
        return self._data.keys()
    
    def __getitem__(self, item):
        return self._data[item]
    
    
    @property
    def faces(self):
        #triangular faces
        if self._3d:
            #raise NotImplementedError('Tetrahedra to faces not implemented yet')
            simp = self.T.simplices
            return np.vstack([simp[:, (0,1,2)],
                               simp[:, (0,3,1)],
                               simp[:, (0, 2, 3)],
                               simp[:, (3, 2, 1)],
                               ])
        else:
            #simplices are already our triangles
            return self.T.simplices
        
    @property
    def face_normals(self):
        # FIXME
        return np.ones([len(self.faces), 3])
    
    @property
    def vertex_normals(self):
        # FIXME
        return np.ones([len(self.vertices), 3])
    
    
    def circumcentres(self):
        cc = np.zeros((len(self.T.simplices), 3))
        for i, s in enumerate(self.T.simplices):
            t_ = self.vertices[s,:]
            
            t0 = t_[0, :]
            #t02 = (t0*t0).sum()
            
            A = t_[1:, :] - t0[None,:]
            b = 0.5*((t_[1:,:] - t0[None, :])**2).sum(1)
            cc[i, :] = np.linalg.solve(A.T,b) + t0
            
        return cc
            
        
        
            
        


@register_module('DelaunayTesselation')
class DelaunayTesselation(ModuleBase):
    input = Input('input')
    output = Output('output')
    
    three_d = Bool(True)

    def execute(self, namespace):
        inp = namespace[self.input]
        
        T = Tesselation(inp, self.three_d)
        try:
            T.mdh = inp.mdh
        except AttributeError:
            pass
    
        namespace[self.output] = T