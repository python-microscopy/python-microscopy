
from .base import register_module, ModuleBase
from .traits import Input, Output, Float, Bool
import numpy as np
from PYME.IO import tabular

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
        raw_fits = tabular.mappingFilter(tabular.recArrayInput(results))
        raw_fits.setMapping('r_curve', '1./(np.abs(A) + np.abs(B) + 1e-6)')  # cap max at 1e6 instead of inf
        raw_fits.mdh = data_source.mdh
        namespace[self.output_fits_raw] = raw_fits

        # filter surfaces and throw out patches with normals that don't point approx. the same way as their neighbors
        results = surfit.filter_quad_results(results, points.T, self.fit_influence_radius, self.alignment_threshold)

        # again, add radius of curvature calculation with lazy evaluation
        filtered_fits = tabular.mappingFilter(tabular.recArrayInput(results.view(surfit.SURF_PATCH_DTYPE_FLAT)))
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
        reconstruction = tabular.mappingFilter({'x': xs, 'y': ys, 'z': zs,
                                                'normal_x': xn, 'normal_y': yn, 'normal_z': zn,
                                                'probe': probe, 'n_points_fit': N, 'patch_id': j,
                                                'r_curve': filtered_fits['r_curve'][j]})
        reconstruction.mdh = data_source.mdh
        namespace[self.output_surface_reconstruction] = reconstruction
