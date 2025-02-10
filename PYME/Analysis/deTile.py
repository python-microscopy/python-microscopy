#!/usr/bin/python

##################
# deTile.py
#
# Copyright David Baddeley, 2009
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
from scipy import fftpack, ndimage
from PYME.Analysis import splitting
# from pylab import ifftshift, ifftn, fftn, fftshift
from numpy.fft import ifftshift, ifftn, fftn, fftshift
from PYME.IO.MetaDataHandler import get_camera_roi_origin
import logging

logger = logging.getLogger(__name__)

def findRectangularROIs(mask):
    #break up any L shaped rois
    #figure(1)
    #imshow(mask)
    #figure(2)
    #plot(np.abs(np.diff(mask, axis=0)).sum(1))
    #print mask.shape
    mask = mask *np.hstack(([1.], np.abs(np.diff(mask, axis=0)).sum(1) == 0))[:, None]

    labels, nLabels = ndimage.label(mask)

    #print nLabels

    rois = []

    #roisizes = np.array([(labels == i).sum() for i in range(1, nLabels +1)])

    for i in range(1, nLabels +1):

        m2 = (labels == i)

        #print m2.sum()
        xc = np.where(m2.sum(1))[0]
        yc = np.where(m2.sum(0))[0]

        if len(xc) > 5 and len(yc) > 5:
            rois.append([xc[0], yc[0], xc[-1], yc[-1]])

    #print xc[0], yc[0], xc[-1], yc[-1]

    return rois


def calcCorrShift(im1, im2):
    im1 = im1 - im1.mean()
    im2 = im2 - im2.mean()
    xc = np.abs(ifftshift(ifftn(fftn(im1)*ifftn(im2))))

    xct = (xc - xc.max()/1.1)*(xc > xc.max()/1.1)
    #print((xct.shape))

    x_ = np.arange(xct.shape[0]) - xct.shape[0]/2
    y_ = np.arange(xct.shape[1]) - xct.shape[1]/2

    xc = xc - xc.min()
    # TODO - subtract half max instead??
    xc = np.maximum(xc - xc.mean(), 0)

    s = xc.sum()

    dx = (xc*x_[:, None]).sum()/s
    dy = (xc*y_[None, :]).sum()/s

    return dx, dy, s

def genDark(ds, mdh, blur=2):
    df = mdh.getEntry('Protocol.DarkFrameRange')
    return ndimage.gaussian_filter(ds[:,:,df[0]:df[1]].mean(2), blur).reshape(list(ds.shape[:2]) + [1,])

def guessFlat(ds, mdh, dark):
    sf = mdh.getEntry('Protocol.DataStartsAt')
    flt = ds[:,:,sf].astype('f')
    n = 1

    for i in range(sf, ds.shape[2]):
        flt +=  ds[:,:,i]
        n+=1

    #print((flt.shape, dark.shape))

    flt = flt/n - dark
    print((flt.shape, flt.mean()))
    return np.minimum(flt.mean()/flt, 2.5)



def assemble_tiles(ds, xps, yps, mdh, split=True, skipMoveFrames=True, shiftfield=None, mixmatrix=[[1.,0.],[0.,1.]], correlate=False, dark=None, flat=None):
    """
    Assemble a tiled image from a dataset ds, given the x and y positions of each frame (xps, yps) and the metadata handler mdh.

    Parameters:
    -----------
    ds : arraylike (3D - x, y, frame)
        The dataset to assemble

    xps : ndarray
        The x position of each frame in nm
    
    yps : ndarray
        The y position of each frame in nm

    mdh : MetaDataHandler
        The metadata handler for the dataset
    
    """
    frameSizeX, frameSizeY, numFrames = ds.shape[:3]

    assert(len(xps) == numFrames)
    assert(len(yps) == numFrames)

    if split:
        frameSizeY /=2
        nchans = 2
        unmux = splitting.Unmixer(shiftfield, mdh.voxelsize_nm.x)
    else:
        nchans = 1

    if mdh.getOrDefault('CameraOrientation.FlipX', False):
        xps = -xps
        logger.warning('Flipping X axis, not well tested (and may invert image)\nIf possible, set stage multiplier instead')
        # TODO - CameraOrientation.FlipX should only be used for a secondary camera (in order to match the orientation of the primary camera)
        # If possible (i.e. primary camera, single camera), it is preferable to configure the stage motion so that it matches the camera using the `multiplier` parameter to `register_piezo()`
        # For secondary cameras we should really flip the image date, not the stage positions here so that the image data is consistent with the primary camera, and so that anything we detect
        # in the tiled image can be mapped back to a stage position.

    if mdh.getOrDefault('CameraOrientation.FlipY', False):
        yps = -yps
        logger.warning('Flipping Y axis, not well tested (and may invert image)\nIf possible, set stage multiplier instead')
        # See notes above for CameraOrientation.FlipX


    #give some room at the edges
    bufSize = 0
    if correlate:
        bufSize = 300
    
    #convert to pixels
    xdp = (bufSize + ((xps - xps.min()) / (mdh.getEntry('voxelsize.x'))).round()).astype('i')
    ydp = (bufSize + ((yps - yps.min()) / (mdh.getEntry('voxelsize.y'))).round()).astype('i')

    #work out how big our tiled image is going to be
    imageSizeX = int(np.ceil(xdp.max() + frameSizeX + bufSize))
    imageSizeY = int(np.ceil(ydp.max() + frameSizeY + bufSize))


    #allocate an empty array for the image
    im = np.zeros([imageSizeX, imageSizeY, nchans])

    # and to record occupancy (to normalise overlapping tiles)
    occupancy = np.zeros([imageSizeX, imageSizeY, nchans])

    #calculate a weighting matrix (to allow feathering at the edges - TODO)
    weights = np.ones((frameSizeX, frameSizeY, nchans))

    edgeRamp = min(100, int(.5*ds.shape[0]))
    weights[:edgeRamp, :, :] *= np.linspace(0,1, edgeRamp)[:,None, None]
    weights[-edgeRamp:, :,:] *= np.linspace(1,0, edgeRamp)[:,None, None]
    weights[:,:edgeRamp,:] *= np.linspace(0,1, edgeRamp)[None, :, None]
    weights[:,-edgeRamp:,:] *= np.linspace(1,0, edgeRamp)[None,:, None]

    
    roi_x0, roi_y0 =get_camera_roi_origin(mdh)
    
    ROIX1 = roi_x0
    ROIY1 = roi_y0

    ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
    ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')

    if dark is None:
        offset = float(mdh.getEntry('Camera.ADOffset'))
    else:
        offset = 0.


    for i in range(mdh.get('Protocol.DataStartsAt', 0), numFrames):
        if xdp[i - 1] == xdp[i] or not skipMoveFrames:
            d = ds[:,:,i].astype('f')
            if not dark is None:
                d = d - dark
            if not flat is None:
                d = d*flat

            if split:
                d = np.concatenate(unmux.Unmix(d, mixmatrix, offset, [ROIX1, ROIY1, ROIX2, ROIY2]), 2)
            #else:
                #d = d.reshape(list(d.shape) + [1])

            imr = (im[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :]/occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :])
            alreadyThere = (weights*occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :]).sum(2) > 0

            #d_ = d.sum(2)

            if split:
                r0 = imr[:,:,0][alreadyThere].sum()
                r1 = imr[:,:,1][alreadyThere].sum()

                if r0 == 0:
                    r0 = 1
                else:
                    r0 = r0/ (d[:,:,0][alreadyThere]).sum()

                if r1 == 0:
                    r1 = 1
                else:
                    r1 = r1/ (d[:,:,1][alreadyThere]).sum()

                rt = np.array([r0, r1])

                imr = imr.sum(2)
            else:
                rt = imr[:,:,0][alreadyThere].sum()
                if rt ==0:
                    rt = 1
                else:
                    rt = rt/ (d[:,:,0][alreadyThere]).sum()

                rt = np.array([rt])

            #print rt

            if correlate:
                if (alreadyThere.sum() > 50):
                    dx = 0
                    dy = 0
                    rois = findRectangularROIs(alreadyThere)

                    w = 0

                    for r in rois:
                        x0,y0,x1,y1 = r
                        #print r
                        dx_, dy_, c_max = calcCorrShift(d.sum(2)[x0:x1, y0:y1], imr[x0:x1, y0:y1].squeeze())
                        print(('d_', dx_, dy_))
                        dx += dx_*c_max
                        dy += dy_*c_max
                        w += c_max
                    
                    dx = int(np.round(dx/w))
                    dy = int(np.round(dy/w))

                    print((dx, dy))

                    #dx, dy = (0,0)
                else:
                    dx, dy = (0,0)

                im[(xdp[i]+dx):(xdp[i]+frameSizeX + dx), (ydp[i] + dy):(ydp[i]+frameSizeY + dy), :] += weights*d#*rt[None, None, :]
                occupancy[(xdp[i] + dx):(xdp[i]+frameSizeX + dx), (ydp[i]+dy):(ydp[i]+frameSizeY + dy), :] += weights
                
            else:
                #print weights.shape, rt.shape, d.shape
                im[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :] += weights*d #*rt[None, None, :]
                occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :] += weights

    ret =  (im/occupancy).squeeze()
    #print ret.shape, occupancy.shape
    ret[occupancy.squeeze() == 0] = 0 #fix up /0s

    return ret



def assemble_tiles_xyztc(data, mdh, correlate=False, dark=None, flat=None):
    """
    Assemble a tiled image from a deterministically tiled data set and the metadata handler mdh.

    Assumes tiles are along the time axis.

    NOTE: This is the most naive and simple implementation, and will put the entire tiled reconstruction and
    a similarly sized occupancy matrix into RAM. There is a good chance that it will blow up on large datasets. 
    TODO - create a pyramid instead.

    Parameters:
    -----------
    ds : arraylike (5D - x, y, z, t, c)
        The dataset to assemble

    mdh : MetaDataHandler
        The metadata handler for the dataset
    
    """
    frameSizeX, frameSizeY, size_z, numTiles, size_c = data.shape

    xps = np.array(mdh['Tiling.XPositions'])
    yps = np.array(mdh['Tiling.YPositions'])

    assert(len(xps) == numTiles)
    assert(len(yps) == numTiles)

    nchans = 1

    if mdh.getOrDefault('CameraOrientation.FlipX', False):
        xps = -xps
        logger.warning('Flipping X axis, not well tested (and may invert image)\nIf possible, set stage multiplier instead')
        # TODO - CameraOrientation.FlipX should only be used for a secondary camera (in order to match the orientation of the primary camera)
        # If possible (i.e. primary camera, single camera), it is preferable to configure the stage motion so that it matches the camera using the `multiplier` parameter to `register_piezo()`
        # For secondary cameras we should really flip the image date, not the stage positions here so that the image data is consistent with the primary camera, and so that anything we detect
        # in the tiled image can be mapped back to a stage position.

    if mdh.getOrDefault('CameraOrientation.FlipY', False):
        yps = -yps
        logger.warning('Flipping Y axis, not well tested (and may invert image)\nIf possible, set stage multiplier instead')
        # See notes above for CameraOrientation.FlipX


    #give some room at the edges
    bufSize = 0
    if correlate:
        bufSize = 300
    
    #convert to pixels
    xdp = (bufSize + ((xps - xps.min()) / (mdh.getEntry('voxelsize.x'))).round()).astype('i')
    ydp = (bufSize + ((yps - yps.min()) / (mdh.getEntry('voxelsize.y'))).round()).astype('i')

    #work out how big our tiled image is going to be
    imageSizeX = int(np.ceil(xdp.max() + frameSizeX + bufSize))
    imageSizeY = int(np.ceil(ydp.max() + frameSizeY + bufSize))


    #allocate an empty array for the image
    im = np.zeros([imageSizeX, imageSizeY, size_z, 1, size_c])

    # and to record occupancy (to normalise overlapping tiles)
    occupancy = np.zeros([imageSizeX, imageSizeY, size_z, 1, size_c])

    #calculate a weighting matrix (to allow feathering at the edges - TODO)
    weights = np.ones((frameSizeX, frameSizeY))

    edgeRamp = min(100, int(.5*frameSizeX))
    weights[:edgeRamp, :] *= np.linspace(0,1, edgeRamp)[:,None]
    weights[-edgeRamp:, :] *= np.linspace(1,0, edgeRamp)[:,None]
    weights[:,:edgeRamp] *= np.linspace(0,1, edgeRamp)[None, :]
    weights[:,-edgeRamp:] *= np.linspace(1,0, edgeRamp)[None,:]

    
    roi_x0, roi_y0 =get_camera_roi_origin(mdh)
    
    ROIX1 = roi_x0
    ROIY1 = roi_y0

    ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
    ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')


    for i in range(numTiles):
        d = data[:,:,:,i:(i+1), :].astype('f')
        
        if not dark is None:
            if np.isscalar(dark):
                d = d - dark
            else:
                d = d - dark[:,:, None, None, None]
        if not flat is None:
            d = d*flat[:,:, None, None, None]


        if correlate:
            #FIXME - This is probably broken!
            raise NotImplementedError('Correlation not implemented for 5D data')
            imr = (im[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :, :, :] 
                / occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :, :, :])
            
            alreadyThere = (weights*occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :, :, :]).sum(2) > 0


            rt = imr[:,:,0][alreadyThere].sum()
            if rt ==0:
                rt = 1
            else:
                rt = rt/ (d[:,:,0][alreadyThere]).sum()

            rt = np.array([rt])
        
            if (alreadyThere.sum() > 50):
                dx = 0
                dy = 0
                rois = findRectangularROIs(alreadyThere)

                w = 0

                for r in rois:
                    x0,y0,x1,y1 = r
                    #print r
                    dx_, dy_, c_max = calcCorrShift(d.sum(2)[x0:x1, y0:y1], imr[x0:x1, y0:y1].squeeze())
                    print(('d_', dx_, dy_))
                    dx += dx_*c_max
                    dy += dy_*c_max
                    w += c_max
                
                dx = int(np.round(dx/w))
                dy = int(np.round(dy/w))

                print((dx, dy))

                #dx, dy = (0,0)
            else:
                dx, dy = (0,0)

            im[(xdp[i]+dx):(xdp[i]+frameSizeX + dx), (ydp[i] + dy):(ydp[i]+frameSizeY + dy), :] += weights*d#*rt[None, None, :]
            occupancy[(xdp[i] + dx):(xdp[i]+frameSizeX + dx), (ydp[i]+dy):(ydp[i]+frameSizeY + dy), :] += weights
            
        else:
            #print weights.shape, rt.shape, d.shape
            im[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :,:,:] += weights[:,:,None,None,None]*d #*rt[None, None, :]
            occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :, :, :] += weights[:,:,None,None,None]

    ret =  (im/occupancy).squeeze()
    #print ret.shape, occupancy.shape
    ret[occupancy.squeeze() == 0] = 0 #fix up /0s

    return ret






