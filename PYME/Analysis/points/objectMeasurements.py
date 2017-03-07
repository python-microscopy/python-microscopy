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


def getIDs(inp, img):
    """

    Parameters
    ----------
    inp: input DataSource, tabular
    img: input image, image.ImageStack object

    Returns
    -------
    ids: Label number from image, mapped to each localization within that label
    numPerObject: Number of localizations within the label that a given localization belongs to

    """
    im_ox, im_oy, im_oz = img.origin

    # account for ROIs
    try:
        p_ox = inp.mdh['Camera.ROIPosX'] * inp.mdh['voxelsize.x'] * 1e3
        p_oy = inp.mdh['Camera.ROIPosY'] * inp.mdh['voxelsize.y'] * 1e3
    except AttributeError:
        raise UserWarning('getIDs requires metadata')

    pixX = np.round((inp['x'] + p_ox - im_ox) / img.pixelSize).astype('i')
    pixY = np.round((inp['y'] + p_oy - im_oy) / img.pixelSize).astype('i')
    pixZ = np.round((inp['z'] - im_oz) / img.sliceSize).astype('i')

    if img.data.shape[2] == 1:
        # disregard z for 2D images
        pixZ = np.zeros_like(pixX)

    ind = (pixX < img.data.shape[0]) * (pixY < img.data.shape[1]) * (pixX >= 0) * (pixY >= 0) * (pixZ >= 0) * (
        pixZ < img.data.shape[2])

    ids = np.zeros_like(pixX)

    # assume there is only one channel
    ids[ind] = np.atleast_3d(img.data[:, :, :, 0].squeeze())[pixX[ind], pixY[ind], pixZ[ind]].astype('i')

    numPerObject, b = np.histogram(ids, np.arange(ids.max() + 1.5) + .5)

    return ids, numPerObject
