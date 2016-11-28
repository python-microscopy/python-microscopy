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
    modes, coeffs, centre = spharm.sphere_expansion(pipeline['x'], pipeline['y'], zscale*pipeline['z'], mmax=mmax)

    print modes, coeffs, centre

    spharm.visualize_reconstruction(modes, coeffs, zscale=1./zscale)


def Plug(visFr):
    visFr.AddMenuItem('Extras', 'Spherical harmonic approximation', lambda e : calc_harmonic_representation(visFr.pipeline))