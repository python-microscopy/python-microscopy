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
    input = Input('input', desc='localizations')
    input_sample_locations = Input('', desc='[optional] - locations to sample density at (if different from localizations')
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
        
        if self.input_sample_locations == '':
            locs = inp
        else:
            locs = namespace[self.input_sample_locations]
        
        N = int(self.n_nearest_neighbours)
        n_ = np.arange(N)
        
        if self.three_d:
            pts = np.vstack([inp['x'], inp['y'], inp['z']]).T
            kdt = cKDTree(pts)
        
            q_pts = np.array([locs['x'], locs['y'], locs['z']]).T
            d_ = np.array([self._dn3d(p, kdt, N, n_) for p in q_pts])
        else:
            pts = np.vstack([inp['x'], inp['y']]).T
            kdt = cKDTree(pts)

            q_pts = np.array([locs['x'], locs['y']]).T
            d_ = np.array([self._dn2d(p, kdt, N, n_) for p in q_pts])
            
        t = tabular.MappingFilter(locs)
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
        #print(self.T.simplices.shape)
        #print(self.vertices.shape)
        
        if self._3d:
            verts = self.vertices
            cc = np.zeros((len(self.T.simplices), 3))
        else:
            verts = self.vertices[:,:2]
            cc = np.zeros((len(self.T.simplices), 2))
            
        #print(verts.shape)
            
        for i, s in enumerate(self.T.simplices):
            t_ = verts[s,:]
            
            t0 = t_[0, :]
            #t02 = (t0*t0).sum()
            
            A = t_[1:, :] - t0[None,:]
            b = 0.5*((t_[1:,:] - t0[None, :])**2).sum(1)
            
            #print(t_.shape, A.shape, b.shape)
            cc[i, :] = np.linalg.solve(A,b) + t0
            
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

@register_module('DelaunayCircumcentres')
class DelaunayCircumcentres(ModuleBase):
    input = Input('input')
    output = Output('output')
    
    append_original_locs = Bool(False)
    
    def execute(self, namespace):
        inp = namespace[self.input]
        
        if not isinstance(inp, Tesselation):
            raise RuntimeError('expected a Tesselation object (as output by the DelaunayTesselation module)')
        
        if inp._3d:
            x, y, z = inp.circumcentres().T
            
            if self.append_original_locs:
                x = np.hstack([x, inp['x']])
                y = np.hstack([y, inp['y']])
                z = np.hstack([z, inp['z']])
            
            pts = {'x' : x, 'y':y, 'z' : z}
        else:
            x, y = inp.circumcentres().T
            
            if self.append_original_locs:
                x = np.hstack([x, inp['x']])
                y = np.hstack([y, inp['y']])
            
            pts = {'x': x, 'y': y, 'z': 0*x}
        
        out = tabular.MappingFilter(pts)
        try:
            out.mdh = inp.mdh
        except AttributeError:
            pass
        
        namespace[self.output] = out


@register_module('Ripleys')
class Ripleys(ModuleBase):
    """ Calculates Ripley's K/L functions for a point set """
    inputPositions = Input('input')
    inputMask = Input('')
    outputName = Output('ripleys')
    normalization = Enum(['K', 'L'])
    nbins = Int(50)
    binSize = Float(50.)
    sampling = Float(5.)
    threaded = Bool(False)
    
    def execute(self, namespace):
        from PYME.Analysis.points import ripleys
        from PYME.IO import MetaDataHandler
        
        points_real = namespace[self.inputPositions]
        mask = namespace.get(self.inputMask, None)
        
        three_d = np.count_nonzero(points_real['z']) > 0
        
        try:
            origin_coords = MetaDataHandler.origin_nm(points_real.mdh)
        except:
            origin_coords = (0, 0, 0)
        
        if three_d:
            bb, K = ripleys.ripleys_k(x=points_real['x'], y=points_real['y'], z=points_real['z'],
                                      mask=mask, n_bins=self.nbins, bin_size=self.binSize,
                                      sampling=self.sampling, threaded=self.threaded, coord_origin=origin_coords)
        else:
            bb, K = ripleys.ripleys_k(x=points_real['x'], y=points_real['y'],
                                      mask=mask, n_bins=self.nbins, bin_size=self.binSize,
                                      sampling=self.sampling, threaded=self.threaded, coord_origin=origin_coords)
        
        if self.normalization == 'L':
            d = 3 if three_d else 2
            bb, L = ripleys.ripleys_l(bb, K, d)
            res = tabular.DictSource({'bins': bb, 'vals': L})
        else:
            res = tabular.DictSource({'bins': bb, 'vals': K})
        
        # propagate metadata, if present
        try:
            res.mdh = points_real.mdh
        except AttributeError:
            pass
        
        namespace[self.outputName] = res