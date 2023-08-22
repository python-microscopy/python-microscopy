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
    print((xct.shape))

    #figure(1)
    #imshow(xct)

    #dx, dy =  ndimage.measurements.center_of_mass(xct)

    #print np.where(xct==xct.max())

    dx, dy = np.where(xct==xct.max())

    return dx[0] - im1.shape[0]/2, dy[0] - im1.shape[1]/2

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



def tile(ds, xm, ym, mdh, split=True, skipMoveFrames=True, shiftfield=None, mixmatrix=[[1.,0.],[0.,1.]], correlate=False, dark=None, flat=None):
    frameSizeX, frameSizeY, numFrames = ds.shape[:3]

    if split:
        frameSizeY /=2
        nchans = 2
        unmux = splitting.Unmixer(shiftfield, mdh.voxelsize_nm.x)
    else:
        nchans = 1

    #x & y positions of each frame
    xps = xm(np.arange(numFrames))
    yps = ym(np.arange(numFrames))

    if mdh.getOrDefault('CameraOrientation.FlipX', False):
        xps = -xps

    if mdh.getOrDefault('CameraOrientation.FlipY', False):
        yps = -yps


    #give some room at the edges
    bufSize = 0
    if correlate:
        bufSize = 300
    
    #convert to pixels
    xdp = (bufSize + ((xps - xps.min()) / (mdh.getEntry('voxelsize.x'))).round()).astype('i')
    ydp = (bufSize + ((yps - yps.min()) / (mdh.getEntry('voxelsize.y'))).round()).astype('i')

    #print (xps - xps.min()), mdh.getEntry('voxelsize.x')

    #work out how big our tiled image is going to be
    imageSizeX = np.ceil(xdp.max() + frameSizeX + bufSize)
    imageSizeY = np.ceil(ydp.max() + frameSizeY + bufSize)

    #print imageSizeX, imageSizeY

    #allocate an empty array for the image
    im = np.zeros([imageSizeX, imageSizeY, nchans])

    # and to record occupancy (to normalise overlapping tiles)
    occupancy = np.zeros([imageSizeX, imageSizeY, nchans])

    #calculate a weighting matrix (to allow feathering at the edges - TODO)
    weights = np.ones((frameSizeX, frameSizeY, nchans))
    #weights[:, :10, :] = 0 #avoid splitter edge artefacts
    #weights[:, -10:, :] = 0

    #print weights[:20, :].shape
    edgeRamp = min(100, int(.5*ds.shape[0]))
    weights[:edgeRamp, :, :] *= np.linspace(0,1, edgeRamp)[:,None, None]
    weights[-edgeRamp:, :,:] *= np.linspace(1,0, edgeRamp)[:,None, None]
    weights[:,:edgeRamp,:] *= np.linspace(0,1, edgeRamp)[None, :, None]
    weights[:,-edgeRamp:,:] *= np.linspace(1,0, edgeRamp)[None,:, None]

    
    roi_x0, roi_y0 =get_camera_roi_origin(mdh)
    
    ROIX1 = roi_x0 + 1
    ROIY1 = roi_y0 + 1

    ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
    ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')

    if dark is None:
        offset = float(mdh.getEntry('Camera.ADOffset'))
    else:
        offset = 0.

#    #get a sorted list of x and y values
#    xvs = list(set(xdp))
#    xvs.sort()
#
#    yvs = list(set(ydp))
#    yvs.sort()

    for i in range(mdh.getEntry('Protocol.DataStartsAt'), numFrames):
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

                    for r in rois:
                        x0,y0,x1,y1 = r
                        #print r
                        dx_, dy_ = calcCorrShift(d.sum(2)[x0:x1, y0:y1], imr[x0:x1, y0:y1])
                        print(('d_', dx_, dy_))
                        dx += dx_
                        dy += dy_
                    
                    dx = np.round(dx/len(rois))
                    dy = np.round(dy/len(rois))

                    print((dx, dy))

                    #dx, dy = (0,0)
                else:
                    dx, dy = (0,0)

                im[(xdp[i]+dx):(xdp[i]+frameSizeX + dx), (ydp[i] + dy):(ydp[i]+frameSizeY + dy), :] += weights*d*rt[None, None, :]
                occupancy[(xdp[i] + dx):(xdp[i]+frameSizeX + dx), (ydp[i]+dy):(ydp[i]+frameSizeY + dy), :] += weights
                
            else:
                #print weights.shape, rt.shape, d.shape
                im[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :] += weights*d #*rt[None, None, :]
                occupancy[xdp[i]:(xdp[i]+frameSizeX), ydp[i]:(ydp[i]+frameSizeY), :] += weights

    ret =  (im/occupancy).squeeze()
    #print ret.shape, occupancy.shape
    ret[occupancy.squeeze() == 0] = 0 #fix up /0s

    return ret






