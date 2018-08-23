
import numpy as np
from PYME.Analysis.points import spherical_harmonics as sh
from PYME.recipes.base import ModuleCollection
from PYME.recipes.localisations import SphericalHarmonicShell
from scipy.special import sph_harm


def _make_dummy_cell(centre=(0., 0., 0.), z_scale=1.):
    # make cube
    x = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
    y = np.array([1, 1, -1, 1, -1, 1, -1, -1], dtype=float)
    z = np.array([1, -1, 1, 1, 1, -1, -1, -1], dtype=float)

    # shift things
    x += centre[0]
    y += centre[1]
    z += centre[2]


    inp = {'x': x, 'y': y, 'z': z}

    recipe = ModuleCollection()
    recipe.namespace['input'] = inp
    shell_module = SphericalHarmonicShell(recipe, input_name='input', z_scale=z_scale, max_m_mode=5, n_iterations=2,
                                          init_tolerance=0.3, output_name='output')
    recipe.add_module(shell_module)
    recipe.execute()
    return recipe.namespace['output']

def test_recipe():
    centre = (0., 0., 0.)
    shell = _make_dummy_cell(centre, 1.)
    # check that this worked OK
    assert shell.mdh['Processing.SphericalHarmonicShell.Centre'] == centre

# ----------------- Tests for PYME.Analysis.points.spherical_harmonics.distance_to_surface -------------------
CENTRE_BASIC = (0., 0., 0.)
MODES_BASIC = [(0, 0)]
COEFFICIENTS_BASIC = [1]
SHIFT = 10.

def test_distance_to_shell_centered():
    min_distance, closest_on_surface = sh.distance_to_surface(CENTRE_BASIC, CENTRE_BASIC, MODES_BASIC,
                                                              COEFFICIENTS_BASIC, d_phi=0.1, z_scale=1.)
    # min_distance from center (0,0,0) should be ~scipy.special.sph_harm() for the (0,0) mode. Since this the 0,0 mode,
    # it won't depend on theta or phi
    np.testing.assert_almost_equal(min_distance, sph_harm(MODES_BASIC[0][0], MODES_BASIC[0][1], 0., 0.), decimal=4)

def test_distance_to_shell_lateral_offset():
    centre_shifted = (SHIFT, 0., 0.)
    min_distance, closest_on_surface = sh.distance_to_surface(centre_shifted, CENTRE_BASIC, MODES_BASIC,
                                                              COEFFICIENTS_BASIC, d_phi=0.1, z_scale=1.)
    np.testing.assert_almost_equal(SHIFT, min_distance + closest_on_surface[0], decimal=4)

    # now the other way
    min_distance, closest_on_surface = sh.distance_to_surface(CENTRE_BASIC, centre_shifted, MODES_BASIC,
                                                              COEFFICIENTS_BASIC, d_phi=0.1, z_scale=1.)
    np.testing.assert_almost_equal(SHIFT, min_distance - closest_on_surface[0], decimal=4)

def test_distance_to_shell_axial_offset():
    # now try it with an offset in z
    centre_shifted = (0., 0., SHIFT)
    min_distance, closest_on_surface = sh.distance_to_surface(centre_shifted, CENTRE_BASIC, MODES_BASIC,
                                                              COEFFICIENTS_BASIC, d_phi=0.1, z_scale=1.)
    np.testing.assert_almost_equal(SHIFT, min_distance + closest_on_surface[2], decimal=4)
    # and the other way
    min_distance, closest_on_surface = sh.distance_to_surface(CENTRE_BASIC, centre_shifted, MODES_BASIC,
                                                              COEFFICIENTS_BASIC, d_phi=0.1, z_scale=1.)
    # allow for worse precision on the subtraction case, especially along z
    np.testing.assert_almost_equal(SHIFT, min_distance - closest_on_surface[2], decimal=2)

# TODO add tests that account for zscale and zscaling with an offset

