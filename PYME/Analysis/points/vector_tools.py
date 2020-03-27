
import numpy as np
import logging

logger = logging.getLogger(__name__)

def spherical_to_cartesian(az, el, r):
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
    # in same notation as sph_harm, az = azimuth, el = zenith
    rsin_zenith = r * np.sin(el)
    x = rsin_zenith * np.cos(az)
    y = rsin_zenith * np.sin(az)
    z = r * np.cos(el)
    return x, y, z

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy, z)
    az = np.arctan2(y, x)
    return az, el, r

def cartesian_to_spherical(x, y, z, azimuth_0=0, zenith_0=0):
    azi, zen, r = cart2sph(x, y, z)
    azimuth = np.mod(azi - azimuth_0, 2*np.pi)
    zenith = np.mod(zen - zenith_0, np.pi)
    return azimuth, zenith, r

def find_principal_axes(x, y, z, sample_fraction=0.5):
    n_points = len(x)
    n_to_sample = int(sample_fraction * n_points)
    index = np.random.choice(range(n_points), n_to_sample, replace=False)
    # calculate principle axes and the spread along them
    eigen_vals, eigen_vecs = np.linalg.eig(np.cov(np.vstack([x[index], y[index], z[index]])))
    standard_deviations = np.sqrt(eigen_vals)
    return standard_deviations, eigen_vecs

def scaled_projection(x, y, z, scaling_factors, scaling_axes):
    # make sure everything is 1D
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
    xp, yp, zp = [np.dot(scaling_factors[ind] * scaling_axes[ind], points) for ind in range(3)]
    # return in original shape
    return xp.reshape(x.shape), yp.reshape(y.shape), zp.reshape(z.shape)

def direction_to_nearest_n_points(x, y, z, x0, y0, z0, n, subsample_fraction=1.):
    """

    Parameters
    ----------
    x
    y
    z
    x0
    y0
    z0
    n
    subsample_fraction: float
        fraction of points to randomly subsample before querying. This can be helpful / necessary when len(x) >= 100,000

    Returns
    -------
    azimuth : ndarray
        azimuth (angle in x,y plane)
    zenith : ndarray
        elevation (angle from z axis)
    r : ndarray
        distance to center of mass for n nearest points for each point queried
    cartesian_vector: ndarray
        Unit vectors pointing to nearest n points for each point queried

    """
    x_0, y_0, z_0 = np.atleast_1d(x0, y0, z0)
    if subsample_fraction < 1:
        n_original = len(x)
        subsample_ind = np.random.choice(n_original, int(n_original * subsample_fraction), replace=False)
        xs, ys, zs = x[subsample_ind], y[subsample_ind], z[subsample_ind]
    else:
        xs, ys, zs = x, y, z


    positions = np.atleast_3d(np.stack([xs, ys, zs])) - np.stack([x_0, y_0, z_0])[:, None, :]
    distances = np.linalg.norm(positions, axis=0)
    I = np.argsort(distances, axis=0)
    # find center of mass of closest n points for each point queried
    x_com, y_com, z_com = xs[I][:n, :].mean(axis=0), ys[I][:n, :].mean(axis=0), zs[I][:n, :].mean(axis=0)
    # get directions to the centers of mass
    azimuth, zenith, r = cartesian_to_spherical(x_com - x_0, y_com - y_0, z_com - z_0)

    v = np.stack([x_com - x_0, y_com - y_0, z_com - z_0])
    cartesian_vector = v / np.linalg.norm(v, axis=0)

    return azimuth, zenith, r, cartesian_vector

def find_points_within_cone(x, y, z, x0, y0, z0, azimuth, zenith, d_omega=0.15, cutoff_r=1500):
    # shift and convert to spherical
    theta, phi, r = cartesian_to_spherical(x - x0, y - y0, z - z0, azimuth, zenith)
    # tilt points along zenith=0 and look at which are within d_omega
    inside_cone = abs(phi - zenith) < d_omega
    inside_cone = np.logical_and(inside_cone, r < cutoff_r)
    return inside_cone

def find_points_within_cylinder(x, y, z, x0, y0, z0, radius, length, v0, v1, v2):
    """

    Parameters
    ----------
    x
    y
    z
    x0
    y0
    z0
    v0 : ndarray
        cartesian vector defining the 'axial axis' of the cylinder
    v1 : ndarray
        cartesian vector orthogonal to the axial axis
    v2 : ndarray
        cartesian vector othogonal to the previous two

    Returns
    -------

    """
    # make sure we have a reasonable task
    assert ((length > 0) and (radius > 0))
    # make sure all input vectors are unit vectors
    v_0 = v0 / np.linalg.norm(v0)
    v_1 = v1 / np.linalg.norm(v1)
    v_2 = v2 / np.linalg.norm(v2)
    # make vectors going from the base/center of the cylinder to all of the points we are checking
    x_c, y_c, z_c = x - x0, y - y0, z - z0
    positions_c = np.stack([x_c, y_c, z_c]).T
    # check if they are within the ends of the cylinder, i.e. if the axial component is positive and less than length
    axial_component = np.dot(positions_c, v_0)
    inside_axially = np.logical_and(np.greater_equal(axial_component, 0.), np.less_equal(axial_component, length))
    # now check if they're inside radially
    radial_component0 = np.dot(positions_c[inside_axially], v_1)
    radial_component1 = np.dot(positions_c[inside_axially], v_2)
    # just in case we're working with a ton of points, throw out the ones which are obviously too far out
    worth_checking = np.logical_and(np.less_equal(radial_component0, radius), np.less_equal(radial_component1, radius))
    # now do the slow bit
    inside_r = np.less_equal(np.sqrt(radial_component0[worth_checking] ** 2 + radial_component1[worth_checking] ** 2), radius)

    # handle the indexing
    inside = np.zeros_like(x, dtype=bool)
    # can't slice because this copies, use where as a sort of disgusting work around
    inside[np.where(inside_axially)[0][np.where(worth_checking)[0]][inside_r]] = True
    return inside, axial_component[inside]


def distance_to_image_mask(mask, points):
    """

    Parameters
    ----------
    mask : PYME.IO.ImageStack
        Binary mask - object label numbers are ignored
    points : PYME.IO.tabular.TabularBase
        points to query distance with respect to mask

    Returns
    -------
    distance : ndarray


    """
    from PYME.IO.MetaDataHandler import get_camera_roi_origin
    from scipy.ndimage import distance_transform_edt

    x0, y0, z0 = mask.origin
    voxel_size = [mask.pixelSize, mask.pixelSize, mask.sliceSize]  # [nm]

    # account for point data ROIs
    try:
        roi_x0, roi_y0 = get_camera_roi_origin(points.mdh)

        p_ox = roi_x0 * points.mdh['voxelsize.x'] * 1e3
        p_oy = roi_y0 * points.mdh['voxelsize.y'] * 1e3
    except AttributeError:
        raise RuntimeError('metadata specifying ROI position and voxelsize are missing')

    # bin points into pixels
    binned_x = np.round((points['x'] + p_ox - x0) / mask.pixelSize).astype('i')
    binned_y = np.round((points['y'] + p_oy - y0) / mask.pixelSize).astype('i')
    binned_z = np.round((points['z'] - z0) / mask.sliceSize).astype('i')

    # FIXME - only works for single color data
    mask_data = np.atleast_3d(mask.data[:,:,:,0].squeeze())
    distance_to_mask = distance_transform_edt(~mask_data, sampling=voxel_size, return_distances=True)

    # now we just sample the distance mask at the indices of the points
    try:
        distances = distance_to_mask[binned_x, binned_y, binned_z]
    except IndexError:
        logger.error('Not all points queried are within image bounds of mask, clipping to image bounds')
        # clip the pixel assignment (there might not be a pixel for each point depending on how mask was generated)
        x_clip = np.clip(binned_x, 0, distance_to_mask.shape[0] - 1)
        y_clip = np.clip(binned_y, 0, distance_to_mask.shape[1] - 1)
        z_clip = np.clip(binned_z, 0, distance_to_mask.shape[2] - 1)
        distances = distance_to_mask[x_clip, y_clip, z_clip]

    return distances

