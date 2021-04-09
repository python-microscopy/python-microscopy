from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr

import numpy as np
from PYME.IO import tabular
import logging


logger = logging.getLogger(__name__)


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
    """
    Ripley's K-function, and alternate normalizations, for examining clustering 
    and dispersion of points within aregion R, where R is defined by a mask (2D 
    or 3D) of the data.
    
        inputPositions : traits.Input
            Localization data source to analyze as PYME.IO.tabular types
        inputMask : traits.Input
            PYME.IO.image.ImageStack mask defining the localization bounding region
        outputName : traits.Output
            Name of resulting PYME.IO.tabular.DictSource data ource
        normalization : traits.Enum
            Ripley's normalization type. See M. A. Kiskowski, J. F. Hancock, 
            and A. K. Kenworthy, "On the use of Ripley's K-function and its 
            derivatives to analyze domain size," Biophys. J., vol. 97, no. 4, 
            pp. 1095-1103, 2009.
        nbins : traits.Int
            Number of bins over which to analyze K-function
        binSize : traits.Float
            K-function bin size in nm
        sampling : traits.Float
            spacing (in nm) of samples from mask / region.
        statistics : traits.Bool
            Monte-Carlo sampling of the structure to determine clustering/
            dispersion probability.
        nsim : int
            Number of Monte-Carlo simulations to run. More simulations = 
            more statistical power. Used if statistics == True.
        significance : float
            Desired significance of 
        threaded : bool
            Calculate pairwise distances using multithreading (faster)
        three_d : bool
            Analyze localizations in 2D or 3D. Requires correct dimensionality
            of input localizations and mask.

    """
    inputPositions = Input('input')
    inputMask = Input('')
    outputName = Output('ripleys')
    normalization = Enum(['K', 'L', 'H', 'dL', 'dH'])
    nbins = Int(50)
    binSize = Float(50.)
    sampling = Float(5.)
    statistics = Bool(False)
    nsim = Int(20)
    significance = Float(0.05)
    threaded = Bool(False)
    three_d = Bool(False)
    
    def execute(self, namespace):
        from PYME.Analysis.points import ripleys
        from PYME.IO import MetaDataHandler
        
        points_real = namespace[self.inputPositions]
        mask = namespace.get(self.inputMask, None)

        # three_d = np.count_nonzero(points_real['z']) > 0
        if self.three_d:
            if np.count_nonzero(points_real['z']) == 0:
                raise RuntimeError('Need a 3D dataset')
            if mask and mask.data.shape[2] < 2:
                raise RuntimeError('Need a 3D mask to run in 3D. Generate a 3D mask or select 2D.')
        else:
            if mask and mask.data.shape[2] > 1:
                raise RuntimeError('Need a 2D mask.')

        if self.statistics and mask is None:
            raise RuntimeError('Mask is needed to calculate statistics.')

        if self.statistics and 1.0/self.nsim > self.significance:
            raise RuntimeError('Need at least {} simulations to achieve a significance of {}'.format(int(np.ceil(1.0/self.significance)),self.significance))
        
        try:
            ox, oy, _ = MetaDataHandler.origin_nm(points_real.mdh)
            origin_coords = (ox, oy, 0)  # see origin_nm docs
        except:
            origin_coords = (0, 0, 0)
        
        if self.three_d:
            bb, K = ripleys.ripleys_k(x=points_real['x'], y=points_real['y'], z=points_real['z'],
                                      mask=mask, n_bins=self.nbins, bin_size=self.binSize,
                                      sampling=self.sampling, threaded=self.threaded, coord_origin=origin_coords)
        else:
            bb, K = ripleys.ripleys_k(x=points_real['x'], y=points_real['y'],
                                      mask=mask, n_bins=self.nbins, bin_size=self.binSize,
                                      sampling=self.sampling, threaded=self.threaded, coord_origin=origin_coords)

        # Run MC simulations
        if self.statistics:
            K_min, K_max, p_clustered, p_dispersed = ripleys.mc_sampling_statistics(K, mask=mask,
                                                                        n_points=len(points_real['x']), n_bins=self.nbins, 
                                                                        three_d=self.three_d, bin_size=self.binSize,
                                                                        significance=self.significance, 
                                                                        n_sim=self.nsim, sampling=self.sampling, 
                                                                        threaded=self.threaded, coord_origin=origin_coords)
        
        # Check for alternate Ripley's normalization
        norm_func = None
        if self.normalization == 'L':
            norm_func = ripleys.ripleys_l
        elif self.normalization == 'dL':
            # Results will be of length 2 less than other results
            norm_func = ripleys.ripleys_dl
        elif self.normalization == 'H':
            norm_func = ripleys.ripleys_h
        elif self.normalization == 'dH':
            # Results will be of length 2 less than other results
            norm_func = ripleys.ripleys_dh
        
        # Apply normalization if present
        if norm_func is not None:
            d = 3 if self.three_d else 2
            bb0, K = norm_func(bb, K, d)  # bb0 in case we use dL/dH
            if self.statistics:
                _, K_min = norm_func(bb, K_min, d)
                _, K_max = norm_func(bb, K_max, d)
                if self.normalization == 'dL' or self.normalization == 'dH':
                    # Truncate p_clustered for dL and dH to match size
                    p_clustered = p_clustered[1:-1]
                    p_dispersed = p_dispersed[1:-1]
            bb = bb0

        if self.statistics:
            res = tabular.DictSource({'bins': bb, 'vals': K, 'min': K_min, 'max': K_max, 'pc': p_clustered, 'pd': p_dispersed})
        else:
            res = tabular.DictSource({'bins': bb, 'vals': K})
        
        # propagate metadata, if present
        try:
            res.mdh = points_real.mdh
        except AttributeError:
            pass
        
        namespace[self.outputName] = res


@register_module('GaussianMixtureModel')
class GaussianMixtureModel(ModuleBase):
    """Fit a Gaussian Mixture to a pointcloud, predicting component membership
    for each input point.

    Parameters
    ----------
    input_points: PYME.IO.tabular
        points to fit. Currently hardcoded to use x, y, and z keys.
    n: Int
        number of Gaussian components in the model for optimization mode `n` 
        and `bayesian`, or maxinum number of components for `bic`
    mode: Enum
        optimization on the number of components. For `n` and `bayesian` the
        GMM uses exactly n components, while for `bic` it is the maximum number
        of components used, with the optimum Bayesian Information Criterion
        used to select the best model.
    covariance: Enum
        type of covariance to use in the model
    label_key: str
        name of membership/label key in output datasource, 'gmm_label' by
        default
    output_labeled: PYME.IO.tabular
        input source with additional column indicating predicted component
        membership of each point
    
    Notes
    -----
    Directly implements or closely wraps scikit-learn mixture.GaussianMixture
    and mixture.BayesianGaussianMixture
    """
    input_points = Input('input')
    n = Int(1)
    mode = Enum(('n', 'bic', 'bayesian'))
    covariance = Enum(('full', 'tied', 'diag', 'spherical'))
    label_key = CStr('gmm_label')
    output_labeled = Output('labeled_points')
    
    def execute(self, namespace):
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
        from PYME.IO import MetaDataHandler

        points = namespace[self.input_points]
        X = np.stack([points['x'], points['y'], points['z']], axis=1)

        if self.mode == 'n':
            gmm = GaussianMixture(n_components=self.n,
                                covariance_type=self.covariance)
            predictions = gmm.fit_predict(X)
        
        elif self.mode == 'bic':
            n_components = range(1, self.n + 1)
            bic = np.zeros(len(n_components))
            for ind in range(len(n_components)):
                gmm = GaussianMixture(n_components=n_components[ind], 
                                    covariance_type=self.covariance)
                gmm.fit(X)
                bic[ind] = gmm.bic(X)
                logger.debug('%d BIC: %f' % (n_components[ind], bic[ind]))

            best = n_components[np.argmin(bic)]
            if best == self.n or (self.n > 10 and best > 0.9 * self.n):
                logger.warning('BIC optimization selected n components near n max')
            
            gmm = GaussianMixture(n_components=best,
                                covariance_type=self.covariance)
            predictions = gmm.fit_predict(X)
        
        elif self.mode == 'bayesian':
            bgm = BayesianGaussianMixture(n_components=self.n,
                                      covariance_type=self.covariance)
            predictions = bgm.fit_predict(X)

        out = tabular.MappingFilter(points)
        try:
            out.mdh = MetaDataHandler.DictMDHandler(points.mdh)
        except AttributeError:
            pass

        out.addColumn(self.label_key, predictions)
        namespace[self.output_labeled] = out
