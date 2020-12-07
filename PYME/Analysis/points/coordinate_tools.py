
import numpy as np
import logging

logger = logging.getLogger(__name__)

def spherical_to_cartesian(az, el, r):
    """
    Convert spherical coordinates into cartesian

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

def find_principle_axes(x, y, z, sample_fraction=None):
    """

    Parameters
    ----------
    x: list-like
        x positions
    y: list-like
        y positions
    z: list-like
        z positions
    sample_fraction: float
        [optional] fraction of points to choose randomly and use for principal axes calculations, reducing computation.
        Default of None uses all points.

    Returns
    -------
    standard_deviations: ndarray
        standard deviations of the input positions along the principle axes
    eigen_vectors: ndarray
        principle axes

    """
    n_points = len(x)

    if sample_fraction is not None:
        n_to_sample = int(sample_fraction * n_points)
        index = np.random.choice(range(n_points), n_to_sample, replace=False)
    else:  # take all
        index = slice(None)

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


def pixel_index_of_points_in_image(image, points):
    """
    Map positions into indices of an image

    Parameters
    ----------
    image: PYME.IO.image.ImageStack
        image with complete metadata
    points: PYME.IO.tabular.TabularBase

    Returns
    -------
    x_index: ndarray
        x pixel index in image for each point
    y_index: ndarray
        y pixel index in image for each point
    z_index: ndarray
        z pixel index in image for each point

    """
    from PYME.IO.MetaDataHandler import origin_nm

    x0, y0, z0 = image.origin

    # account for point data ROIs
    p_ox, p_oy, p_oz = origin_nm(points.mdh)

    # Image origin is referenced to top-left corner of pixelated image.
    # FIXME - localisations are currently referenced to centre of raw pixels
    x_index = np.floor((points['x'] + p_ox - x0) / image.voxelsize_nm.x).astype('i')
    y_index = np.floor((points['y'] + p_oy - y0) / image.voxelsize_nm.y).astype('i')
    z_index = np.floor((points['z'] + p_oz - z0) / image.voxelsize_nm.z).astype('i')

    return x_index, y_index, z_index


def distance_to_image_mask(mask, points):
    """
    Calculate the distance from point positions to the edge of an image mask.

    Parameters
    ----------
    mask : PYME.IO.ImageStack
        Binary mask where True denotes inside of the mask. Can be integers too, e.g. from an image of labels where 0
        denotes unlabeled, but this will be compressed into a single mask of inside object(s) and outside. Edge of the
        mask is considered the first pixel which is False.
    points : PYME.IO.tabular.TabularBase
        points to query distance with respect to mask

    Returns
    -------
    distance : ndarray
        Distance from each point to the edge of the mask in units of nanometers. Negative values denote being inside
        of the mask.


    """
    from scipy.ndimage import distance_transform_edt, binary_dilation

    binned_x, binned_y, binned_z = pixel_index_of_points_in_image(mask, points)

    # FIXME - only works for single color data
    mask_data = np.atleast_3d(mask.data[:,:,:,0].squeeze())

    # calculate (and negate) distances from inside the mask (True) to edge of mask (where it turns False)
    distance_to_mask = - distance_transform_edt(mask_data, sampling=mask.voxelsize_nm, return_distances=True)
    # add distances going outward from the mask edge
    distance_to_mask = distance_to_mask + distance_transform_edt(1 - binary_dilation(mask_data),
                                                                 sampling=mask.voxelsize_nm, return_distances=True)
    # clip the pixel assignment (there might not be a pixel for each point depending on how mask was generated)
    if np.any(np.stack((binned_x, binned_y, binned_z)) < 0):
        logger.error('Not all points queried are within image bounds of mask, clipping to image bounds')
        x_clip = np.clip(binned_x, 0, distance_to_mask.shape[0] - 1)
        y_clip = np.clip(binned_y, 0, distance_to_mask.shape[1] - 1)
        z_clip = np.clip(binned_z, 0, distance_to_mask.shape[2] - 1)
        distances = distance_to_mask[x_clip, y_clip, z_clip]
    else:
        # now we just sample the distance mask at the indices of the points
        distances = distance_to_mask[binned_x, binned_y, binned_z]

    return distances

