"""
Estimate spherical harmonics from a point data set
"""
import numpy as np
from scipy.special import sph_harm
from scipy import linalg

def sph2cart(az, el, r):
    """
    Convert sperical coordinates into cartesian

    Parameters
    ----------
    az : ndarray
        azimuth (angle in x,y plane)
    el : ndarray
        elevation (angle from z axis)
    r : ndarray
        radius

    Returns
    -------

    x, y, z

    """
    # in same notation as sph_harm, az = theta, el = phi
    rsin_phi = r * np.sin(el)
    x = rsin_phi * np.cos(az)
    y = rsin_phi * np.sin(az)
    z = r * np.cos(el)
    return x, y, z

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy, z)
    az = np.arctan2(y, x)
    return az, el, r

def r_sph_harm(m, n, theta, phi):
    """
    return real valued spherical harmonics. Uses the convention that m > 0 corresponds to the cosine terms, m < zero the
    sine terms

    Parameters
    ----------
    m : int

    n : int

    theta : ndarray
        the azimuth angle in [0, 2pi]
    phi : ndarray
        the elevation in [0, pi]

    Returns
    -------

    """
    if m > 0:
        return (1./np.sqrt(2)*(-1)**m)*sph_harm(m, n, theta, phi).real
    elif m == 0:
        return sph_harm(m, n, theta, phi).real
    else:
        return (1./np.sqrt(2)*(-1)**m)*sph_harm(m, n, theta, phi).imag


def sphere_expansion(x, y, z, mmax=3, centre_points=True):
    """
    Project coordinates onto spherical harmonics

    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    z : ndarray
        z coordinates
    mmax : int
        Maximum order to calculate to
    centre_points : bool
        Subtract the mean from the co-ordinates before projecting

    Returns
    -------

    modes : list of tuples
        a list of the (m, n) modes projected onto
    c : ndarray
        the mode coefficients
    centre : tuple
        the x, y, z centre of the object (if we centred the points pripr to calculation).


    """
    if centre_points:
        x0, y0, z0 = x.mean(), y.mean(), z.mean()
    else:
        x0, y0, z0 = 0, 0, 0# x.mean(), y.mean(), z.mean()

    x_, y_, z_ = x - x0, y - y0, z - z0

    theta, phi, r = cart2sph(x_, y_, z_)

    A = []
    modes = []
    for m in range(mmax + 1):
        for n in range(-m, m + 1):
            sp_mode = r_sph_harm(n, m, theta, phi)
            A.append(sp_mode)

            modes.append((m, n))

    A = np.vstack(A)

    c = linalg.lstsq(A.T, r)[0]

    return modes, c, (x0, y0, z0)

def sphere_expansion_clean(x, y, z, mmax=3, centre_points=True, nIters = 2, tol_init = 0.3):
    """
    Project coordinates onto spherical harmonics

    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    z : ndarray
        z coordinates
    mmax : int
        Maximum order to calculate to
    centre_points : bool
        Subtract the mean from the co-ordinates before projecting

    Returns
    -------

    modes : list of tuples
        a list of the (m, n) modes projected onto
    c : ndarray
        the mode coefficients
    centre : tuple
        the x, y, z centre of the object (if we centred the points prior to calculation).


    """
    if centre_points:
        x0, y0, z0 = x.mean(), y.mean(), z.mean()
    else:
        x0, y0, z0 = 0, 0, 0# x.mean(), y.mean(), z.mean()

    x_, y_, z_ = x - x0, y - y0, z - z0

    theta, phi, r = cart2sph(x_, y_, z_)

    A = []
    modes = []
    for m in range(mmax + 1):
        for n in range(-m, m + 1):
            sp_mode = r_sph_harm(n, m, theta, phi)
            A.append(sp_mode)

            modes.append((m, n))

    A = np.vstack(A).T

    tol = tol_init

    c = linalg.lstsq(A, r)[0]

    #recompute, discarding outliers
    for i in range(nIters):
        pred = np.dot(A, c)
        error = abs(r - pred)/r
        mask = error < tol
        #print mask.sum(), len(mask)

        c = linalg.lstsq(A[mask,:], r[mask])[0]
        tol /=2

    return modes, c, (x0, y0, z0)

def reconstruct_from_modes(modes, coeffs, theta, phi):
    r_ = 0

    for (m, n), c in zip(modes, coeffs):
        r_ += c * (r_sph_harm(n, m, theta, phi))

    return r_


def visualize_reconstruction(modes, coeffs, d_phi=.1, zscale=1.0):
    from mayavi import mlab
    phi, theta = np.mgrid[0:(np.pi + d_phi):d_phi, 0:(2*np.pi + d_phi):d_phi]

    r = reconstruct_from_modes(modes, coeffs, theta, phi)
    x1, y1, z1 = sph2cart(theta, phi, r)

    mlab.figure()
    mlab.mesh(x1, y1, z1*zscale)

def distance_to_surface(position, centre, modes, coeffs, d_phi=0.1, z_scale=5.):
    """

    Parameters
    ----------
    position : list-like of ndarrays
        Arrays of positions to query (in cartesian coordinates), i.e. [np.array(x), np.array(y), np.array(z)]
    centre : iterable
        Center of the spherical harmonic shell
    modes : list
        List of (m, n) mode tuples
    coeffs : list
        List of coefficients corresponding to the list of modes
    d_phi : float
        Sets the step size in radians of theta and phi arrays used in reconstructing the spherical harmonic shell
    z_scale : float
        The scale parameter used multiplicatively on the z-positions when fitting the spherical harmonics to the
        original point cloud.

    Returns
    -------
    min_distance : float
        minimum distance from 'position' (i.e. input coordinate) to the spherical harmonic surface
    closest_point_on_surface : tuple of floats
        returns the position in cartesian coordinates of the point on the surface closest to the input 'position'

    """
    x, y, z = position
    n_points = len(x)
    phi, theta = np.mgrid[0:(np.pi + d_phi):d_phi, 0:(2 * np.pi + d_phi):d_phi]

    r = reconstruct_from_modes(modes, coeffs, theta, phi)
    x_shell, y_shell, z_shell = sph2cart(theta, phi, r)

    # center the position we're querying, remembering that the shell needs to be scaled inversely scaling in the fit
    x = x[:] - centre[0]  # perform an implicit copy so we don't shift the input
    y = y[:] - centre[1]
    z = z[:] - centre[2]/z_scale

    # calculate the distance between all our points and the shell, remembering needs to be scaled inversely scaling in
    # the fit
    dist = np.sqrt((x - x_shell[:,:,None]) ** 2 + (y - y_shell[:,:,None]) ** 2 + ((z - z_shell[:,:,None]/z_scale) ** 2))

    # unfortunately cannot currently specify two axes for numpy.argmin, so we'll have to flatten the first two dims
    n_shell_coords = dist.shape[0] * dist.shape[1]
    dist_flat = dist.reshape((n_shell_coords, n_points))
    min_ind = np.argmin(dist_flat, axis=0)

    p_ind = range(n_points)
    return dist_flat[min_ind[p_ind], p_ind], (x_shell.reshape(n_shell_coords)[min_ind],
                                              y_shell.reshape(n_shell_coords)[min_ind],
                                              z_shell.reshape(n_shell_coords)[min_ind])

