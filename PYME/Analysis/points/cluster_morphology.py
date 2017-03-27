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

def get_labels_from_image(label_image, points):
    """
    Function to extract labels from a segmented image (2D or 3D) at given locations. 

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
    im_ox, im_oy, im_oz = label_image.origin

    # account for ROIs
    try:
        p_ox = points.mdh['Camera.ROIPosX'] * points.mdh['voxelsize.x'] * 1e3
        p_oy = points.mdh['Camera.ROIPosY'] * points.mdh['voxelsize.y'] * 1e3
    except AttributeError:
        raise RuntimeError('label image requires metadata specifying the voxelsize')

    pixX = np.round((points['x'] + p_ox - im_ox) / label_image.pixelSize).astype('i')
    pixY = np.round((points['y'] + p_oy - im_oy) / label_image.pixelSize).astype('i')
    pixZ = np.round((points['z'] - im_oz) / label_image.sliceSize).astype('i')

    label_data = label_image.data

    if label_data.shape[2] == 1:
        # disregard z for 2D images
        pixZ = np.zeros_like(pixX)

    ind = (pixX < label_data.shape[0]) * (pixY < label_data.shape[1]) * (pixX >= 0) * (pixY >= 0) * (pixZ >= 0) * (
        pixZ < label_data.shape[2])

    ids = np.zeros_like(pixX)

    # assume there is only one channel
    ids[ind] = np.atleast_3d(label_data[:, :, :, 0].squeeze())[pixX[ind], pixY[ind], pixZ[ind]].astype('i')

    numPerObject, b = np.histogram(ids, np.arange(ids.max() + 1.5) + .5)

    return ids, numPerObject

measurement_dtype = [('count', '<i4'),
                     ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                     ('gyrationRadius', '<f4'),
                     ('axis0i', '<f4'), ('axis0j', '<f4'), ('axis0k', '<f4'),
                     ('axis1i', '<f4'), ('axis1j', '<f4'), ('axis1k', '<f4'),
                     ('axis2i', '<f4'), ('axis2j', '<f4'), ('axis2k', '<f4'),
                     ('sigma0', '<f4'), ('sigma1', '<f4'), ('sigma2', '<f4'),
                     ('theta', '<f4'), ('phi', '<f4')]

def measure_3d(x, y, z, output=None):
    if output is None:
        output = np.zeros(1, measurement_dtype)
    
    #count
    N = len(x)
    output['count'] = N
    
    #centroid
    xc, yc, zc = x.mean(), y.mean(), z.mean()
    
    output['x'] = xc
    output['y'] = yc
    output['z'] = zc
    
    #find mean-subtracted points
    x_, y_, z_ = x - xc, y - yc, z - zc
    
    #radius of gyration
    output['gyrationRadius'] = np.sqrt(np.mean(x_*x_ + y_*y_ + z_*z_))

    #principle axes
    u, s, v = np.linalg.svd(np.vstack([x_, y_, z_]).T)

    try:
        for i in range(3):
            output['axis%di' % i], output['axis%dj' % i], output['axis%dk' % i] = v[i]
            #std. deviation along axes
            output['sigma%d' % i] = s[i]/np.sqrt(N-1)
    except IndexError:  # this occurs if e.g. the cluster is only 2 points, and only has two principle axes
        # zero all svd outputs
        for i in range(3):
            output['axis%di' % i], output['axis%dj' % i], output['axis%dk' % i], output['sigma%d' % i] = 0, 0, 0, 0
            output['theta'], output['phi'] = 0, 0
        return output
    
    pa = v[0]
    #angle of principle axis
    output['theta'] = np.arctan(pa[0]/pa[1])
    output['phi'] = np.arcsin(pa[2])
    
    #TODO - compactness based on pairwise distances?
    
    return output
    