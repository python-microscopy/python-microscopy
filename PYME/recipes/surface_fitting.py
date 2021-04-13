
from .base import register_module, ModuleBase
from .traits import Input, Output, Float, Int, Bool, CStr, ListFloat
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
    remesh = Bool(False)
    
    def execute(self, namespace):
        #from PYME.experimental import dual_marching_cubes_v2 as dual_marching_cubes
        from PYME.experimental import dual_marching_cubes
        # from PYME.experimental import triangle_mesh
        from PYME.experimental import _triangle_mesh as triangle_mesh
        
        dmc = dual_marching_cubes.PiecewiseDualMarchingCubes(self.threshold_density)
        dmc.set_octree(namespace[self.input].truncate_at_n_points(int(self.n_points_min)))
        tris = dmc.march(dual_march=False)

        print('Generating TriangularMesh object')
        surf = triangle_mesh.TriangleMesh.from_np_stl(tris, smooth_curvature=self.smooth_curvature)
        
        print('Generated TriangularMesh object')
        
        if self.repair:
            surf.repair()
            
        if self.remesh:
            # target_length = np.mean(surf._halfedges[''][surf._halfedges['length'] != -1])
            surf.remesh(5, l=0.5, n_relax=10)

        namespace[self.output] = surf


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

    output_name = Output('harmonic_shell')
    output_name_mapped = Output('shell_mapped')


    def execute(self, namespace):
        from PYME.Analysis.points import spherical_harmonics

        points = tabular.MappingFilter(namespace[self.input_name])

        shell = spherical_harmonics.ScaledShell()
        shell.set_fitting_points(points['x'], points['y'], points['z'])
        shell.fit_shell(max_iterations=self.max_iterations, tol_init=self.init_tolerance)

        separations, closest_points = shell.distance_to_shell((points['x'], points['y'], points['z']),
                                                              d_angles=self.d_angles)

        points.addColumn(self.name_distance_to_shell, separations)
        points.addColumn(self.name_inside_shell, shell.check_inside())

        namespace[self.output_name] = shell
        namespace[self.output_name_mapped] = points


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
