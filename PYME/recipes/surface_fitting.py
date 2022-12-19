
from .base import register_module, ModuleBase
from .traits import Input, Output, Float, Int, Bool, CStr, ListFloat, Enum
import numpy as np
from PYME.IO import tabular
import logging


logger = logging.getLogger(__name__)

@register_module('FitSurfaceWithPatches')
class FitSurfaceWithPatches(ModuleBase):
    """
    Fits patches of quadratic planes to point data.

    Parameters
    ----------

    input : Input
        tabular data containing x, y, and z columns for the surface to be fit to.
    fit_influence_radius : Float
        The region around each localization to include in the surface fit [nm]. The fit is performed on all points
        falling within this radius of each control point.
    reconstruction_radius : Float
        The size of the reconstructed surface patch. This should usually be <= fit_influence_radius.
    constrain_surface_to_point : Bool
        Whether the fit should be constrained to pass through the control point.
    limit_reconstruction_to_support_hull: Bool
        If enabled, this will clip each surface reconstruction to the convex-hull of all the points used for the fit.
        Useful for avoiding the generation of large surface patches from isolated antibodies, but also reduces the
        ability to paper-over holes.
    alignment_threshold : Float
        Lower alignment_threshold keeps more patches, higher rejects more. For each patch, normal vectors for adjacent
        patches (within fit_influence_radius) are projected along the normal vector. If the median of these dot products
         is less than the alignment_threshold, the patch is removed.
    reconstruction_point_spacing : Float
        Spacing of points used to reconstruct the surface.

    Returns
    -------

    out_fits_raw : Output
        tabular data containing fit results for quadratic surface patches following the form
        $w(u,v) = A \times u^2 + B \times v^2$. See PYME.Analysis.points.surfit.fit_quad_surf.
    out_fits_filtered : Output
        Filtered version of out_fits_filtered. Use alignment_threshold to reject surface patches which differ greatly in
        orientation from their neighbors.
    out_surface_reconstruction : Output
        Reconstruction of out_fits_filtered. Fitted points are augmented with additional points to interpolate the
        surface. The density of this reconstruction can be altered using the reconstruction_point_spacing and
        reconstruction_radius parameters. Each point contains normal information and is compatible with shaded-point
        viewing.

    """
    input = Input('localizations')

    fit_influence_radius = Float(100, desc=('The region around each localization to include in the surface fit [nm]. ' 
                                            'The fit is performed on all points falling within this radius of each '
                                            'control point'))
    reconstruction_radius = Float(50, desc=('The size of the reconstructed surface patch. This should usually '
                                            'be <= fit_influence_radius'))
    constrain_surface_to_point = Bool(True,
                                      desc='Whether the fit should be constrained to pass through the control point')
    limit_reconstruction_to_support_hull = Bool(False,
                                                desc='If enabled, this will clip each surface reconstruction to the '
                                                     'convex-hull of all the points used for the fit. Useful for '
                                                     'avoiding the generation of large surface patches from isolated '
                                                     'antibodies, but also reduces the ability to paper-over holes')
    alignment_threshold = Float(0.85, desc='Lower alignment_threshold keeps more patches, higher rejects more.'
                                           'For each patch, normal vectors for adjacent patches (within '
                                           'fit_influence_radius) are projected along the normal vector. If the median '
                                           'of these dot products is less than the alignment_threshold, the patch is '
                                           'removed.')
    reconstruction_point_spacing = Float(10., desc='Spacing of points used to reconstruct the surface')

    output_fits_raw = Output('raw_surface_fits')
    output_fits_filtered = Output('filtered_surface_fits')
    output_surface_reconstruction = Output('surface_reconstruction')

    def execute(self, namespace):
        from PYME.Analysis.points import surfit
        data_source = namespace[self.input]

        # arrange point data in the format we expect
        points = np.vstack([data_source['x'].astype('f'), data_source['y'].astype('f'), data_source['z'].astype('f')])

        # do the actual fitting - this fits one surface for every point in the dataset
        results = surfit.fit_quad_surfaces_Pr(points.T, self.fit_influence_radius,
                                              fitPos=(not self.constrain_surface_to_point))

        # calculate a radius of curvature from our polynomials
        raw_fits = tabular.MappingFilter(tabular.RecArraySource(results))
        raw_fits.setMapping('r_curve', '1./(np.abs(A) + np.abs(B) + 1e-6)')  # cap max at 1e6 instead of inf
        raw_fits.mdh = data_source.mdh
        namespace[self.output_fits_raw] = raw_fits

        # filter surfaces and throw out patches with normals that don't point approx. the same way as their neighbors
        results = surfit.filter_quad_results(results, points.T, self.fit_influence_radius, self.alignment_threshold)

        # again, add radius of curvature calculation with lazy evaluation
        filtered_fits = tabular.MappingFilter(tabular.RecArraySource(results.view(surfit.SURF_PATCH_DTYPE_FLAT)))
        filtered_fits.setMapping('r_curve', '1./(np.abs(A) + np.abs(B) + 1e-6)')
        filtered_fits.mdh = data_source.mdh
        namespace[self.output_fits_filtered] = filtered_fits

        # reconstruct the surface by generating an augmented point data set for each surface, adding virtual
        # localizations spread evenly across each surface patch. Note this is done on the filtered fits.
        if self.limit_reconstruction_to_support_hull:
            xs, ys, zs, xn, yn, zn, N, j = surfit.reconstruct_quad_surfaces_Pr_region_cropped(results,
                                                                                              self.reconstruction_radius,
                                                                                              points.T,
                                                                                              fit_radius=self.fit_influence_radius,
                                                                                              step=self.reconstruction_point_spacing)
        else:
            xs, ys, zs, xn, yn, zn, N, j = surfit.reconstruct_quad_surfaces_Pr(results, self.reconstruction_radius,
                                                                               step=self.reconstruction_point_spacing)

        j = j.astype(int)
        try:
            # duck-type probe; note we assume surface was fit to single-color data
            probe = np.ones_like(xs) * data_source['probe'][0]
        except KeyError:
            probe = np.zeros_like(xs)
        # construct a new datasource with our augmented points
        reconstruction = tabular.MappingFilter({'x': xs, 'y': ys, 'z': zs,
                                                'xn': xn, 'yn': yn, 'zn': zn,
                                                'probe': probe, 'n_points_fit': N, 'patch_id': j,
                                                'r_curve': filtered_fits['r_curve'][j]})
        reconstruction.mdh = data_source.mdh
        namespace[self.output_surface_reconstruction] = reconstruction


@register_module('DualMarchingCubes')
class DualMarchingCubes(ModuleBase):
    input = Input('octree')
    output = Output('mesh')
    
    threshold_density = Float(2e-5)
    n_points_min = Int(50) # lets us truncate on SNR
    
    smooth_curvature = Bool(True)  # TODO: This is actually a mesh property, so it can be toggled outside of the recipe.
    repair = Bool(False)
    remesh = Bool(True)
    cull_inner_surfaces = Bool(True)
    
    def execute(self, namespace):
        #from PYME.experimental import dual_marching_cubes_v2 as dual_marching_cubes
        from PYME.experimental import dual_marching_cubes
        # from PYME.experimental import triangle_mesh
        from PYME.experimental import _triangle_mesh as triangle_mesh
        from PYME.IO import MetaDataHandler

        inp = namespace[self.input]
        md = MetaDataHandler.DictMDHandler(getattr(inp, 'mdh', None)) # get metadata from the input dataset if present
        
        dmc = dual_marching_cubes.PiecewiseDualMarchingCubes(self.threshold_density)
        dmc.set_octree(inp.truncate_at_n_points(int(self.n_points_min)))
        tris = dmc.march(dual_march=False)

        print('Generating TriangularMesh object')
        surf = triangle_mesh.TriangleMesh.from_np_stl(tris, smooth_curvature=self.smooth_curvature)
        
        print('Generated TriangularMesh object')
        
        if self.repair:
            surf.repair()
            
        if self.remesh:
            # target_length = np.mean(surf._halfedges[''][surf._halfedges['length'] != -1])
            surf.remesh(5, l=0.5, n_relax=10)
            
        if self.cull_inner_surfaces:
            surf.remove_inner_surfaces()

        self._params_to_metadata(md)
        surf.mdh = md #inject metadata
        
        namespace[self.output] = surf

@register_module('Isosurface')
class Isosurface(ModuleBase):
    input = Input('input')
    output = Output('mesh')

    threshold = Float(0.5)
    remesh = Bool(True)
    
    def execute(self, namespace):
        from PYME.experimental import isosurface

        im = namespace[self.input]
        T = isosurface.isosurface(im.data_xyztc[:,:,:,0,0].astype('f'), isolevel=self.threshold, voxel_size=im.voxelsize, origin=im.origin, remesh=self.remesh)

        namespace[self.output] = T

@register_module('DelaunayMarchingTetrahedra')
class DelaunayMarchingTetrahedra(ModuleBase):
    input = Input('delaunay0')
    output = Output('mesh')
    
    threshold_density = Float(2e-5)

    repair = Bool(False)
    remesh = Bool(False)

    def execute(self, namespace):
        #from PYME.experimental import dual_marching_cubes_v2 as dual_marching_cubes
        from PYME.experimental import marching_tetrahedra
        from PYME.experimental import _triangle_mesh as triangle_mesh
        import time

        simplices = namespace[self.input].T.simplices
        vertices = namespace[self.input].T.points[simplices]
        values = namespace[self.input].dn[simplices]
        
        
        mt = marching_tetrahedra.MarchingTetrahedra(vertices, values, self.threshold_density)
        print('Marching...')
        start = time.time()
        tris = mt.march()
        stop = time.time()
        elapsed = stop-start
        print('Generated mesh in {} s'.format(elapsed))
        print('Generating TriangularMesh object')
        surf = triangle_mesh.TriangleMesh.from_np_stl(tris)
        print('Generated TriangularMesh object')

        if self.repair:
            surf.repair()
            
        if self.remesh:
            #target_length = np.mean(surf._halfedges['length'][surf._halfedges['length'] != -1])
            surf.remesh(5, l=0.5, n_relax=10)
        
        namespace[self.output] = surf

@register_module('DistanceToMesh')
class DistanceToMesh(ModuleBase):
    input_mesh = Input('mesh')
    input_points = Input('input')
    output = Output('output')

    def execute(self, namespace):
        from PYME.IO import tabular
        from PYME.experimental.isosurface import distance_to_mesh

        inp = namespace[self.input_points]
        surf = namespace[self.input_mesh]

        points = np.vstack([inp['x'], inp['y'], inp['z']]).T

        d = distance_to_mesh(points, surf)

        out = tabular.MappingFilter(inp)
        out.addColumn('distance_to_{}'.format(self.input_mesh), d)

        try:
            out.mdh = inp.mdh
        except AttributeError:
            pass

        namespace[self.output] = out

@register_module('FilterMeshComponentsByVolume')
class FilterMeshComponentsByVolume(ModuleBase):
    """
    USE WITH CAUTION - Will likely change/dissapear in future versions without deprecation.
    
    Create a new mesh which only contains components within a given size range
    
    Notes
    ----- 
    - this is extremely specific (arguably too specific for it's own recipe module), it would be good to incorporate the functionality 
      in a more generic mesh filtering/manipulation module in the future.
    - this would be better positioned (along with some of the others here) in, e.g., a `mesh.py` or `meshes.py` top level set of recipes, rather than `surface_fitting`  
    
    """
    input_mesh = Input('mesh')
    min_size = Float(100.0)
    max_size = Float(1e9)
    output = Output('filtered_mesh')

    def execute(self, namespace):
        from PYME.experimental import _triangle_mesh as triangle_mesh

        mesh = triangle_mesh.TriangleMesh(mesh=namespace[self.input_mesh])
        mesh.keep_components_by_volume(self.min_size, self.max_size)

        namespace[self.output] = mesh

@register_module('SphericalHarmonicShell')
class SphericalHarmonicShell(ModuleBase):
    """
    Fits a shell represented by a series of spherical harmonic co-ordinates to a 3D set of points. The points
    should represent a hollow, fairly round structure (e.g. the surface of a cell nucleus). The object should NOT
    be filled (i.e. points should only be on the surface).

    Parameters
    ----------
    input_name: PYME.IO.tabular.TabularBase
        input localizations to fit a shell to
    max_n_mode: Int
        maximum order of spherical harmonics to use in fit
    max_iterations: Int
        number of fit iterations
    init_tolerance: Float
        relative outlier tolerance. Used to ignore outliers in subsequent iterations
    d_angles: Float
        Sets the step size in radians of zenith and azimuth arrays used in reconstructing the spherical harmonic shell.
        Only relevant for distance_to_shell column of mapped output.
    bound_tolerance: Float
        Factor of fitting point spatial extent to allow in each dimension. The
        default of 0 means this tolerance is not checked, otherwise if the
        spatial extent of shell is a factor of `bound_tolerance` larger than 
        extent of the fitting points, the fit is failed.


    Returns
    ------
    output_name: PYME.Analysis.points.spherical_harmonics.ScaledShell
        The shell instance, with to_hdf method for use with PYME.recipes.outputs.HDFOutput
    output_name_mapped: PYME.IO.tabular.TabularBase
        localizations used to fit the shell, with two additional columns corresponding to whether than point is inside
        the shell, and it's approximate distance to the shell (the latter being subject to the precision of a gridded
        reconstruction subject to `d_angles` input).
    name_inside_shell: CStr
        name of column in `output_name_mapped` indicating whether each point is inside of the shell or not
    name_distance_to_shell: CStr
        name of column in `output_name_mapped` containing the approximate distance to the shell, in the same units as
        the input localizations [nm]

    """
    input_name = Input('input')

    max_n_mode = Int(3)
    max_iterations = Int(2)
    init_tolerance = Float(0.3, desc='Fractional tolerance on radius used in first iteration')

    name_inside_shell = CStr('inside_shell')
    name_distance_to_shell = CStr('distance_to_shell')
    d_angles = Float(0.1)
    bound_tolerance = Float(0.0, desc='Factor of fitting point image bounds to allow.')

    output_name = Output('harmonic_shell')
    output_name_mapped = Output('shell_mapped')


    def execute(self, namespace):
        from PYME.Analysis.points import spherical_harmonics

        points = tabular.MappingFilter(namespace[self.input_name])

        shell = spherical_harmonics.ScaledShell()
        x = points['x'].astype(np.float32, copy=False)
        y = points['y'].astype(np.float32, copy=False)
        z = points['z'].astype(np.float32, copy=False)
        shell.set_fitting_points(x, y, z)
        shell.fit_shell(max_iterations=self.max_iterations, tol_init=self.init_tolerance)

        separations, closest_points = shell.distance_to_shell((x, y, z),
                                                              d_angles=self.d_angles)
        if self.bound_tolerance != 0:
            shell_bounds = shell.approximate_image_bounds()
            max_extent = np.asarray(shell._fitting_point_bounds.extent) * self.bound_tolerance
            if np.any(shell_bounds.extent > max_extent):
                raise AssertionError('Shell bounds exceed fit-point bound tolerance')
            
        points.addColumn(self.name_distance_to_shell, separations)
        points.addColumn(self.name_inside_shell, shell.check_inside())

        # add median distance to shell as a goodness of fit parameter
        shell.table_representation.addColumn('median_residual', np.array([np.median(separations),]))
        namespace[self.output_name] = shell
        namespace[self.output_name_mapped] = points


@register_module('AddSphericalHarmonicShellMappedCoords')
class AddSphericalHarmonicShellMappedCoords(ModuleBase):
    """Add scaled spherical coordinates and harmonic shell radius to an input
    tabular datasource, returning a copy
    
    TODO: Rename and/or refactor
    
    """
    input_localizations = Input('input')
    input_shell = Input('harmonic_shell')

    name_scaled_azimuth = CStr('scaled_azimuth')
    name_scaled_zenith = CStr('scaled_zenith')
    name_scaled_radius = CStr('scaled_radius')
    name_normalized_radius = CStr('normalized_radius')

    output_mapped = Output('harmonic_shell_mapped')


    def execute(self, namespace):
        from PYME.Analysis.points import spherical_harmonics
        from PYME.IO import MetaDataHandler

        inp = namespace[self.input_localizations]
        points = tabular.MappingFilter(inp)
        shell = namespace[self.input_shell]
        if isinstance(shell, tabular.TabularBase):
            shell = spherical_harmonics.ScaledShell.from_tabular(shell)
        
        # map points to scaled spherical coordinates
        azimuth, zenith, r = shell.shell_coordinates((points['x'].astype(np.float32, copy=False),
                                                      points['y'].astype(np.float32, copy=False),
                                                      points['z'].astype(np.float32, copy=False)))
        
        # lookup shell radius at those angles
        r_shell = spherical_harmonics.reconstruct_shell(shell.modes,
                                                        shell.coefficients,
                                                        azimuth, zenith)

        points.addColumn(self.name_scaled_azimuth, azimuth)
        points.addColumn(self.name_scaled_zenith, zenith)
        points.addColumn(self.name_scaled_radius, r)
        points.addColumn(self.name_normalized_radius, r / r_shell)
        
        try:
            points.mdh = MetaDataHandler.DictMDHandler(inp.mdh)
        except AttributeError:
            pass
        namespace[self.output_mapped] = points

@register_module('SHShellRadiusDensityEstimate')
class SHShellRadiusDensityEstimate(ModuleBase):
    """
    Estimate the normalized radius histogram for a uniform distribution within
    the shell.
    TODO: make more generic re: SDF / other surfaces

    Also estimates the volume of the shell, the anisotropy of the shell, and
    the standard deviation along its principle axes (if filled with a uniform
    distribution)
    
    TODO - rename/refactor. This has too much logic in the recipe module itself, it also feels very special case and the name might make it hard to discover.
    """
    input_shell = Input('harmonic_shell')

    r_bin_spacing = Float(0.05)
    sampling_nm = ListFloat([75, 75, 75]) # TODO - convert this to a single float - we only use the first entry
    jitter_iterations = Int(3)
    batch_size = Int(100000)
    sampling_method = Enum(['grid', 'uniform random'])

    output = Output('r_uniform_kde')
    shell_properties_output = Output('')


    def execute(self, namespace):
        from PYME.Analysis.points import spherical_harmonics
        from PYME.IO import MetaDataHandler, tabular

        shell = namespace[self.input_shell]
        if isinstance(shell, tabular.TabularBase):
            shell = spherical_harmonics.ScaledShell.from_tabular(shell)

        if self.sampling_method == 'uniform random':
            # uniform random sampling with Monte-Carlo rejection.
            bin_edges, counts = shell.uniform_random_radial_density(n_radial_bins=int(1./self.r_bin_spacing), batch_size=self.batch_size, target_sampling_nm=self.sampling_nm[0])
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            # estimate the volume, convert from nm^3 to um^3
            # note, volume estimate will have an error of around +- sqrt(N_counts)
            volume = counts.sum() * (np.prod(self.sampling_nm) / (1e9))

            res = tabular.DictSource({
                'bin_centers': bin_centers,
                'counts': counts
            })
            
            res.mdh = MetaDataHandler.DictMDHandler(getattr(shell, 'mdh', None))

            

            # FIXME - this is really gross - we must not pass data / results in metadata
            # Leaving here for now, for backwards compatibility, but needs to be fixed / removed somewhat urgently
            # if we record sampling in the metadata, as we should, the volume calculation could easily be done in the consuming module)      
            res.mdh['SHShellRadiusDensityEstimate.Volume'] = float(volume)


            # record module parameters  - FIXME this should be under the 'Analysis' top-level metadata key.
            res.mdh['SHShellRadiusDensityEstimate.sampling_nm'] = self.sampling_nm[0] 

            namespace[self.output] = res

            if self.shell_properties_output != '':
                props = tabular.tabular.MappingFilter(shell.table_representation)
                props.addColumn('radial_dens_volume', np.array([float(volume)]))
                namespace[self.shell_properties_output] 
        
            return

        ## old code  (sampling on a regular grid) TODO - refactor out of this module          
        
        bin_edges = np.arange(0, 1.0 + self.r_bin_spacing, self.r_bin_spacing, 
                              dtype=np.float32)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        out_hist = np.zeros(len(bin_centers), dtype=np.float32)

        # get shell bounds, make grid within
        shell_bounds = shell.approximate_image_bounds()
        xv = np.arange(shell_bounds.x0, shell_bounds.x1 + self.sampling_nm[0],
                       self.sampling_nm[0], dtype=np.float32)
        yv = np.arange(shell_bounds.y0, shell_bounds.y1 + self.sampling_nm[1],
                       self.sampling_nm[1], dtype=np.float32)
        zv = np.arange(shell_bounds.z0, shell_bounds.z1 + self.sampling_nm[2],
                       self.sampling_nm[2], dtype=np.float32)
        # meshgrid with copy false to use a view (don't write x,y,z after)
        x, y, z = np.meshgrid(xv, yv, zv, indexing='ij', copy=False)

        v_estimates = []
        sdev_estimates = []
        
        for _ in range(self.jitter_iterations):
            xr = np.random.rand(len(xv), len(yv), len(zv)).astype(np.float32, copy=False)
            yr = np.random.rand(len(xv), len(yv), len(zv)).astype(np.float32, copy=False)
            zr = np.random.rand(len(xv), len(yv), len(zv)).astype(np.float32, copy=False)
            xr = (xr - 0.5) * self.sampling_nm[0] + x
            yr = (yr - 0.5) * self.sampling_nm[1] + y
            zr = (zr - 0.5) * self.sampling_nm[2] + z
            azi, zen, r = shell.shell_coordinates((xr, yr, zr))
            r_shell = spherical_harmonics.reconstruct_shell(shell.modes,
                                                            shell.coefficients,
                                                            azi, zen)
            inside = r < r_shell
            N = np.sum(inside)
            # normalize r, and writeover to save memory
            r = r[inside] / r_shell[inside]
            # sum-normalize this iteration and add to output
            out_hist += np.histogram(r, bins=bin_edges)[0] / N

            # record volume estimate
            v_estimates.append(N)
            # estimate spread along principle axes of the shell
            X = np.vstack([xr[inside], yr[inside], zr[inside]])
            if N > self.batch_size:
                # downsample to avoid memory error
                X = X[:, np.random.choice(N, self.batch_size, replace=False)]
            # TODO - do we need to be mean-centered?
            X = X - X.mean(axis=1)[:, None]
            _, s, _ = np.linalg.svd(X.T)
            # svd cov is not normalized, handle that
            sdev_estimates.append(s / np.sqrt(X.shape[1] - 1))  # with bessel's correction
        
        # finish the average
        out_hist = out_hist / self.jitter_iterations
        # finish the volume calculation, convert from nm^3 to um^3
        volume = np.mean(v_estimates) * (np.prod(self.sampling_nm) / (1e9))
        # average the standard deviation estimates
        standard_deviations = np.mean(np.stack(sdev_estimates), axis=0)
        # similar to Basser, P. J., et al. doi.org/10.1006/jmrb.1996.0086
        # note that singular values are square roots of the eigenvalues. Use 
        # the sample standard deviation rather than pop.
        anisotropy = np.sqrt(np.var(standard_deviations**2, ddof=1)) / (np.sqrt(3) * np.mean(standard_deviations**2))
        
        res = tabular.DictSource({
            'bin_centers': bin_centers,
            'counts': out_hist
        })
        try:
            res.mdh = MetaDataHandler.DictMDHandler(shell.mdh)
        except AttributeError:
            pass
        

        if self.shell_properties_output != '':
            props = tabular.tabular.MappingFilter(shell.table_representation)
            props.addColumn('radial_dens_volume', np.array([float(volume)]))
            props.addColumn('radial_dens_std_dev', np.array([standard_deviations.astype('f4')]))
            props.addColumn('radial_dens_anisotropy', np.array([float(anisotropy)]))
            namespace[self.shell_properties_output] 

        namespace[self.output] = res


@register_module('ImageMaskFromSphericalHarmonicShell')
class ImageMaskFromSphericalHarmonicShell(ModuleBase):
    """

    Parameters
    ----------
    input_shell: spherical_harmonics.ScaledShell()
        input localizations to fit a shell to
    bounds_source: PYME.IO.tabular
        optional input to estimate image bounds from, otherwise the points
        used to fit the shell are used
    voxelsize_nm: list
        x, y, z pixel size in nm


    Returns
    ------
    output: PYME.IO.image.ImageStack
        boolean mask True inside, False outside
    """
    input_shell = Input('harmonic_shell')
    input_image_bound_source = Input('input')
    voxelsize_nm = ListFloat([75, 75, 75])
    output = Output('output')


    def execute(self, namespace):
        from PYME.IO.image import ImageBounds, ImageStack
        from PYME.IO.MetaDataHandler import DictMDHandler, origin_nm

        shell = namespace[self.input_shell]
        if isinstance(shell, tabular.TabularBase):
            from PYME.Analysis.points import spherical_harmonics
            shell = spherical_harmonics.ScaledShell.from_tabular(shell)
        image_bound_source = namespace[self.input_image_bound_source]
        # TODO - make bounds estimation more generic - e.g. to match an existing image.
        b = ImageBounds.estimateFromSource(image_bound_source)
        ox, oy, _ = origin_nm(image_bound_source.mdh)
        
        nx = np.ceil((np.ceil(b.x1) - np.floor(b.x0)) / self.voxelsize_nm[0]) + 1
        ny = np.ceil((np.ceil(b.y1) - np.floor(b.y0)) / self.voxelsize_nm[1]) + 1
        nz = np.ceil((np.ceil(b.z1) - np.floor(b.z0)) / self.voxelsize_nm[2]) + 1
        
        x = np.arange(np.floor(b.x0), b.x0 + nx * self.voxelsize_nm[0], self.voxelsize_nm[0])
        y = np.arange(np.floor(b.y0), b.y0 + ny * self.voxelsize_nm[1], self.voxelsize_nm[1])
        z = np.arange(np.floor(b.z0), b.z0 + nz * self.voxelsize_nm[2], self.voxelsize_nm[2])
        logger.debug('mask size %s' % ((len(x), len(y), len(z)),))

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        inside = shell.check_inside(xx.ravel(), yy.ravel(), zz.ravel())
        inside = np.reshape(inside, xx.shape)
        
        mdh = DictMDHandler({
            'voxelsize.x': self.voxelsize_nm[0] / 1e3,
            'voxelsize.y': self.voxelsize_nm[1] / 1e3,
            'voxelsize.z': self.voxelsize_nm[2] / 1e3,
            'ImageBounds.x0': x.min(), 'ImageBounds.x1': x.max(),
            'ImageBounds.y0': y.min(), 'ImageBounds.y1': y.max(),
            'ImageBounds.z0': z.min(), 'ImageBounds.z1': z.max(),
            'Origin.x': ox + b.x0,
            'Origin.y': oy + b.y0,
            'Origin.z': b.z0
        })

        namespace[self.output] = ImageStack(data=inside, mdh=mdh, 
                                            haveGUI=False)
