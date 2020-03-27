
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, Float, CStr, Int
from PYME.IO import tabular


@register_module('SphericalHarmonicShell')
class SphericalHarmonicShell(ModuleBase):
    """
    Fits a shell represented by a series of spherical harmonic co-ordinates to a 3D set of points. The points
    should represent a hollow, fairly round structure (e.g. the surface of a cell nucleus). The object should NOT
    be filled (i.e. points should only be on the surface).

    Parameters
    ----------

        FIXME

    Inputs
    ------
        FIXME


    """
    input_name = Input('input')

    max_m_mode = Int(3)
    n_iterations = Int(2)
    init_tolerance = Float(0.3, desc='Fractional tolerance on radius used in first iteration')

    name_inside_shell = CStr('inside_shell')
    name_distance_to_shell = CStr('distance_to_shell')
    d_zenith = Float(0.1)

    output_name = Output('harmonic_shell')
    output_name_mapped = Output('shell_mapped')


    def execute(self, namespace):
        from PYME.Analysis.points.spherical_harmonics import fitting

        points = tabular.MappingFilter(namespace[self.input_name])

        shell = fitting.ScaledShell()
        shell.set_fitting_points(points['x'], points['y'], points['z'])
        shell.fit_shell(max_m_mode=self.max_m_mode, n_iterations=self.n_iterations, tol_init=self.init_tolerance)

        separations, closest_points = shell.distance_to_shell((points['x'], points['y'], points['z']),
                                                              d_zenith=self.d_zenith)

        # TODO - CALCULATE DISPLACEMENT VECTOR FOR EACH POINT?
        # TODO CALCULATE R_NORM AND USE IT TO ONLY CALCULATE DISPLACEMENT VECTOR USING POINTS INSIDE NUCLEUS
        points.addColumn(self.name_distance_to_shell, separations)
        points.addColumn(self.name_inside_shell, shell.check_inside())

        namespace[self.output_name] = shell
        namespace[self.output_name_mapped] = points

