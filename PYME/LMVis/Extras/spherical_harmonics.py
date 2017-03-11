import PYME.Analysis.points.spherical_harmonics as spharm



def calc_harmonic_representation(pipeline, mmax=5, zscale=5.0):
    """

    Parameters
    ----------
    pipeline : tabular
        tabular object containing point data
    mmax : int
        maximum order to fit to
    zscale : float
        scaling factor in z to make flat nuclei appear more spherical (and hence fit with a smaller number of modes)

    Returns
    -------

    """
    from mayavi import mlab
    modes, coeffs, centre = spharm.sphere_expansion_clean(pipeline['x'], pipeline['y'], zscale*pipeline['z'], mmax=mmax)

    #print modes, coeffs, centre

    spharm.visualize_reconstruction(modes, coeffs, zscale=1./zscale)
    mlab.points3d(pipeline['x'] - centre[0], pipeline['y'] - centre[1], pipeline['z'] - centre[2]/zscale, mode='point')


def Plug(visFr):
    visFr.AddMenuItem('Extras', 'Spherical harmonic approximation', lambda e : calc_harmonic_representation(visFr.pipeline))