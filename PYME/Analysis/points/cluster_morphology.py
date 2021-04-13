#!/usr/bin/python
##################
# objectMeasurements.py
#
# Copyright David Baddeley, Andrew Barentine 2017
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import numpy as np

def get_labels_from_image(label_image, points, minimum_localizations=1):
    """
    Function to extract labels from a segmented image (2D or 3D) at given locations.
    
    TODO - Move this somewhere more sensible and drop minimum_localizations in favour of post-filtering.

    Parameters
    ----------
    label_image: PYME.IO.image.ImageStack instance
        an image containing object labels
    points: tabular-like (PYME.IO.tabular, np.recarray, pandas DataFrame) containing 'x', 'y' & 'z' columns
        locations at which to extract labels

    Returns
    -------
    ids: Label number from image, mapped to each localization within that label
    numPerObject: Number of localizations within the label that a given localization belongs to

    """
    from PYME.Analysis.points.coordinate_tools import pixel_index_of_points_in_image

    pixX, pixY, pixZ = pixel_index_of_points_in_image(label_image, points)

    label_data = label_image.data

    if label_data.shape[2] == 1:
        # disregard z for 2D images
        pixZ = np.zeros_like(pixX)

    ind = (pixX < label_data.shape[0]) * (pixY < label_data.shape[1]) * (pixX >= 0) * (pixY >= 0) * (pixZ >= 0) * (
        pixZ < label_data.shape[2])

    ids = np.zeros_like(pixX)

    # assume there is only one channel
    ids[ind] = np.atleast_3d(label_data[:, :, :, 0].squeeze())[pixX[ind], pixY[ind], pixZ[ind]].astype('i')

    # check if we keep all labels
    if minimum_localizations > 1:  # skip if we don't need this
        labels, counts = np.unique(ids, return_counts=True)
        labels, counts = labels[1:], counts[1:]  # ignore unlabeled points, or zero-label
        for label, count in zip(labels, counts):
            if count < minimum_localizations:
                ids[ids == label] = 0

    numPerObject, b = np.histogram(ids, np.arange(ids.max() + 1.5) + .5)

    return ids, numPerObject

measurement_dtype = [('count', '<i4'),
                     ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                     ('gyrationRadius', '<f4'),
                     ('median_abs_deviation', '<f4'),
                     ('axis0', '<3f4'), ('axis1', '<3f4'), ('axis2', '<3f4'),
                     ('sigma0', '<f4'), ('sigma1', '<f4'), ('sigma2', '<f4'),
                     ('sigma_x', '<f4'), ('sigma_y', '<f4'), ('sigma_z', '<f4'),
                     ('anisotropy', '<f4'),
                     ('theta', '<f4'), ('phi', '<f4')]

def measure_3d(x, y, z, output=None):
    """
    Calculates various metrics for a single cluster, whose 3D coordinates are input. 'output', an optional input
    argument, allows one to use a single structured array to contain measures for multiple clusters. See
    PYME.recipes.localisations.MeasureClusters3D

    Parameters
    ----------
    x : ndarray
        x-positions, typically in nanometers
    y : ndarray
        y-positions, typically in nanometers
    z : ndarray
        z-positions, typically in nanometers
    output : dict-like
        If present, output will have measurements written into it, otherwise a structured array will be created and
        returned

    Returns
    -------
    output : dict-like, structured ndarray
        dict-like object, typically a structured ndarray containing the following measurements:
        count : int
            Number of localizations (points) in the cluster
        x : float
            x center of mass
        y : float
            y center of mass
        z : float
            z center of mass
        gyrationRadius : float
            root mean square displacement to center of cluster, a measure of compaction or spatial extent see also
            supplemental text of DOI: 10.1038/nature16496
        median_abs_deviation : float
            median absolute deviation, i.e. median of the radius from the 
            median (x, y, z) position of the cluster
        axis0 : ndarray, shape (3,)
            principle axis which accounts for the largest variance of the cluster, i.e. corresponds to the largest
            eigenvalue
        axis1 : ndarray, shape (3,)
            next principle axis
        axis2 : ndarray, shape (3,)
            principle axis corresponding to the smallest eigenvalue
        sigma0 : float
            standard deviation along axis0
        sigma1 : float
            standard deviation along axis1
        sigma2 : float
            standard deviation along axis2
        anisotropy : float
            metric of anisotropy based on the spread along principle axes. Standard deviations of alpha * [1, 0, 0],
            where alpha is a scalar, will result in an 'anisotropy' value of 1, i.e. maximally anisotropic. Completely
            isotropic clusters will have equal standard deviations, i.e. alpha * [1, 1, 1], which corresponds to an
            'anisotropy' value of 0. Intermediate cases result in values between 0 and 1.
        theta : float
            Azimuthal angle, in radians, along which the principle axis (axis0) points
        phi : float
            Zenith angle, in radians, along which the principle axis (axis0) points

    """
    if output is None:
        output = np.zeros(1, measurement_dtype)
    
    #count
    N = len(x)
    output['count'] = N
    if N < 3:
        import warnings
        warnings.warn('measure_3D can only be used on clusters of size 3 or larger', UserWarning)
        return
    
    #centroid
    xc, yc, zc = x.mean(), y.mean(), z.mean()
    
    output['x'] = xc
    output['y'] = yc
    output['z'] = zc
    
    #find mean-subtracted points
    x_c, y_c, z_c = x - xc, y - yc, z - zc
    xx_c = x_c * x_c
    yy_c = y_c * y_c
    zz_c = z_c * z_c
    # calculate standard deviations along cartesian coords, including bessel's correction
    output['sigma_x'] = np.sqrt(xx_c.sum() / (N - 1))
    output['sigma_y'] = np.sqrt(yy_c.sum() / (N - 1))
    output['sigma_z'] = np.sqrt(zz_c.sum() / (N - 1))
    
    #radius of gyration and median absolute deviation
    output['gyrationRadius'] = np.sqrt(np.mean(xx_c + yy_c + zz_c))
    output['median_abs_deviation'] = np.median(np.sqrt(
        (x - np.median(x)) ** 2 + (y - np.median(y)) ** 2 + (z - np.median(z)) ** 2
    ))

    #principle axes
    u, s, v = np.linalg.svd(np.vstack([x_c, y_c, z_c]).T)

    standard_deviations = s / np.sqrt(N - 1)  # with bessel's correction
    for i in range(3):
        output['axis%d' % i] = v[i]
        # std. deviation along axes
        output['sigma%d' % i] = standard_deviations[i]

    # similar to Basser, P. J., et al. doi.org/10.1006/jmrb.1996.0086
    # note that singular values are square roots of the eigenvalues. Use the sample standard deviation rather than pop.
    output['anisotropy'] = np.sqrt(np.var(standard_deviations**2, ddof=1)) / (np.sqrt(3) * np.mean(standard_deviations**2))
    
    pa = v[0]
    #angle of principle axis
    output['theta'] = np.arctan(pa[0]/pa[1])
    output['phi'] = np.arcsin(pa[2])
    
    #TODO - compactness based on pairwise distances?
    
    return output
    
