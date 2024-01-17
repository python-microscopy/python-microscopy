#!/usr/bin/python

##################
# extractImages.py
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
from PYME.IO import tabular
import scipy.ndimage
import logging
logger = logging.getLogger(__name__)

import PYME.warnings


def getPSFSlice(datasource, resultsSource, metadata, zm=None):
    f1 = tabular.ResultsFilter(resultsSource, error_x=[1, 30], A=[10, 500], sig=(150 / 2.35, 900 / 2.35))

    ims, pts, zvals, zis = extractIms(datasource, f1, metadata, zm)
    return getPSF(ims, pts, zvals, zis)

#def getPointPoss(dataSource, zm, zmid):
#    for i in range()

def extractIms(dataSource, results, metadata, zm =None, roiSize=10, nmax = 1000):
    ims = np.zeros((2*roiSize, 2*roiSize, len(results['x'])))
    
    vs = metadata.voxelsize_nm
    points = (np.array([results['x']/vs.x,
                        results['y']/vs.y,
                        results['A']]).T)

    pts = np.round(points[:,:2])
    points[:,:2] = points[:,:2] - pts
    ts = results['tIndex']
    bs = results['fitResults_background']

    ind = (pts[:,0] > roiSize)*(pts[:,1] > roiSize)*(pts[:,0] < (dataSource.shape[0] - roiSize))*(pts[:,1] < (dataSource.shape[1] - roiSize))

    #print ind.sum()

    points = points[ind,:]
    pts = pts[ind,:]
    ts = ts[ind]
    bs = bs[ind]

    if not zm is None:
        zvals = np.array(list(set(zm.yvals)))
        zvals.sort()

        zv = zm(ts.astype('f'))
        #print zvals
        #print zv

        zis = np.array([np.argmin(np.abs(zvals - z)) for z in zv])
        #print zis
    else:
        zvals = np.array([0])
        zis = 0.*ts

    for i in range(len(ts)):
        x = pts[i,0]
        y = pts[i,1]

        t = ts[i]
        #print t

        ims[:,:,i] = dataSource[(x-roiSize):(x+roiSize), (y-roiSize):(y+roiSize), t].squeeze() - bs[i]

    return ims - metadata['Camera.ADOffset'], points, zvals, zis


def getPSF(ims, points, zvals, zis):
    height, width = ims.shape[0],ims.shape[1]
    kx,ky = np.mgrid[:height,:width]#,:self.sliceShape[2]]

    kx = np.fft.fftshift(kx - height/2.)/height
    ky = np.fft.fftshift(ky - width/2.)/width
    

    d = np.zeros((height, width, len(zvals)))
    print((d.shape))

    for i in range(len(points)):
        F = np.fft.fftn(ims[:,:,i])
        p = points[i,:]
        #print zis[i]
        #print ifftn(F*exp(-2j*pi*(kx*-p[0] + ky*-p[1]))).real.shape
        d[:,:,zis[i]] = d[:,:,zis[i]] + np.fft.ifftn(F*np.exp(-2j*np.pi*(kx*-p[0] + ky*-p[1]))).real

    d = len(zvals)*d/(points[:,2].sum())
    
    #estimate background as a function of z by averaging rim pixels
    bg = (d[0,:,:].squeeze().mean(0) + d[-1,:,:].squeeze().mean(0) + d[:,0,:].squeeze().mean(0) + d[:,-1,:].squeeze().mean(0))/4
    d = d - bg[None,None,:]
    d = d/d.sum(1).sum(0)[None,None,:]

    return d

def getIntCenter(im):
    im = im.squeeze()
    X, Y, Z = np.ogrid[0:im.shape[0], 0:im.shape[1], 0:im.shape[2]]

    #from pylab import *
    #imshow(im.max(2))

    X = X.astype('f') - X.mean()
    Y = Y.astype('f') - Y.mean()
    Z = Z.astype('f')

    im2 = im - im.min()
    im2 = im2 - 0.5*im2.max()
    im2 = im2*(im2 > 0)

    ims = im2.sum()

    x = (im2*X).sum()/ims
    y = (im2*Y).sum()/ims
    z = (im2*Z).sum()/ims

    #print x, y, z

    return x, y, z

def _expand_z(ps_shape, im_shape, points):
    """
    Expand ROI size in z to maximum supported by image and point locations.
    
    Parameters
    ----------
    ps_shape : 3-tuple of psf half roi sizes
    im_shape : 3-tuple giving
    points : locations of points

    Returns
    -------

    new_shape : 3-tuple of expanded PSF shape
    """
    
    sx, sy, sz = ps_shape
    logger.debug('Expanding PSF z size to maximum supported by data')
    
    pts = np.asarray(points)
    dzl = np.min(pts[:,2]) - 1
    dzh = np.min(im_shape[2] - pts[:,2]) -1
    
    szn = int(min(dzl, dzh))
    
    if szn < sz:
        logger.warning('New PSF is SMALLER than requested (szn = %d, request = %d)' % (szn, sz))
    
    return sx, sy, szn


def getPSF3D(im, points, PSshape=(30,30,30), blur=(0.5, 0.5, 1), normalize=True, centreZ=True, centreXY=True,
             x_offset=0, y_offset=0, z_offset=0,expand_z=False, pad=(0,0,4)):
    
    """
    Extract a 3D PSF by averaging the images of a set of point sources.

    Parameters
    ----------
    im : ndarray
        3D image stack
    points : ndarray of shape (N, 3)
        array of point locations
    PSshape : tuple of 3 ints
        half size of PSF in pixels
    blur : tuple of 3 floats
        standard deviation of Gaussian blur to apply to each dimension
    normalize : bool, 'max', or 'sum'
        If True, or max normalize the entire PSF to the maximum value. If 'sum', normalize to the sum of the in-focus plane of the PSF
    centreZ : bool
        If True, make sure the PSF is centred in z. If False, individual PSF images will still be aligned with respect to each other, but the
        PSF as a whole will not be. Useful when taking multi-channel PSFs where you want to preserve chromatic shifts between the channels
        in the PSF so that deconvolution will take care of chromatic shifts without an explicit separate shift correction step. Also used
        when extracting multi-channel PSFs for Biplane localization.
    centreXY : bool
        Similar to centreZ, but for the xy plane. Do not use for biplane PSFs (it is generally not possible to assume a constant
        lateral shift across the field of view, so this needs to be handled with shiftmaps).
    expand_z : bool
        Expand to the entire z range of the image. Was useful for deconvolution to ensure PSF sizes match the image size, but
        largely unnecessary as deconvolution code now has PSF expansion built in.
    pad : tuple of 3 ints
        How much to pad the PSF in each dimension to avoid wrap-around when performing Fourier domain shifting. Default is (0,0,2)
    """
    
    # pad the PSF shape to avoid wrap-around when performing Fourier domain shifting
    PSshape = np.array(PSshape) + np.array(pad)
    
    if expand_z:
        PSshape = _expand_z(PSshape, im.shape, points)
        
    sx, sy, sz = PSshape
    height, width, depth = 2 * np.array(PSshape, dtype=int) + 1

    kx, ky, kz = np.mgrid[:height, :width, :depth]  # ,:self.sliceShape[2]]

    kx = np.fft.fftshift(kx - height / 2.) / height
    ky = np.fft.fftshift(ky - width / 2.) / width
    kz = np.fft.fftshift(kz - depth / 2.) / depth

    logger.debug('Extracting PSF of size: %d, %d, %d [px]' % (height, width, depth))
    d = np.zeros((height, width, depth))

    imgs = []
    dxs = []
    dys = []
    dzs = []

    for px,py,pz in points:
        print((px, py, pz))
        px = int(px)
        py = int(py)
        pz = int(pz)
        imi = im[(px-sx):(px+sx+1),(py-sy):(py+sy+1),(pz-sz):(pz+sz+1)]
        print((imi.shape))
        dx, dy, dz = getIntCenter(imi)
        dz -= sz

        imgs.append(imi)
        dxs.append(dx)
        dys.append(dy)
        dzs.append(dz)

    dxs = np.array(dxs)
    dys = np.array(dys)
    dzs = np.array(dzs)

    dzm = dzs.mean()
    dxm = dxs.mean()
    dym = dys.mean()

    if not centreZ:
        #images will still be aligned with each other, but any shift in the channel as a whole will be maintained.
        dzs = dzs - dzm + z_offset
        dzm = 0

    if not centreXY:
        dxs = dxs - dxm + x_offset
        dys = dys - dym + y_offset
        dzm = 0

    for imi, dx, dy, dz in zip(imgs, list(dxs), list(dys), list(dzs)):
        F = np.fft.fftn(imi)
        d = d + np.fft.ifftn(F*np.exp(-2j*np.pi*(kx*-dx + ky*-dy + kz*-dz))).real

    print('dzs:', dzs)
    if np.any(np.abs(dzs) > pad[2]):
        PYME.warnings.warn('Axial shift of PSF is greater than padding. PSF is likely to exhibit wrap-around artifacts. This is \
                           typically caused by insufficent z-extent in the requested PSF shape and/or the calibration z-stack.')
    
    # remove padding
    px, py, pz = pad
    d= d[px:(-px if (px > 0) else None), py:(-py if py > 0 else None), pz:(-pz if pz > 0 else None)]

    d = scipy.ndimage.gaussian_filter(d, blur)
    #estimate background as a function of z by averaging rim pixels
    #bg = (d[0,:,:].squeeze().mean(0) + d[-1,:,:].squeeze().mean(0) + d[:,0,:].squeeze().mean(0) + d[:,-1,:].squeeze().mean(0))/4
    d = d - d.min()

    if (normalize == True) or (normalize == 'max'):
        d = d/d.max()
    elif normalize == 'sum':
        d = d/d[:,:,d.shape[2]/2].sum()

    return d, (dxm, dym, dzm)
    
def backgroundCorrectPSFWF(d):
    import numpy as np
    from scipy import linalg
    
    zf = int(d.shape[2]/2)
        
    #subtract a linear background in x
    Ax = np.vstack([np.ones(d.shape[0]), np.arange(d.shape[0])]).T        
    bgxf = (d[0,:,zf] + d[-1,:,zf])/2
    gx = linalg.lstsq(Ax, bgxf)[0]
    
    d = d - np.dot(Ax, gx)[:,None,None]
    
    #do the same in y
    Ay = np.vstack([np.ones(d.shape[1]), np.arange(d.shape[1])]).T        
    bgyf = (d[:,0,zf] + d[:,-1,zf])/2
    gy = linalg.lstsq(Ay, bgyf)[0]
    
    d = d - np.dot(Ay, gy)[None, :,None]
    
    
    #estimate background on central slice as mean of rim pixels
    #bgr = (d[0,:,zf].mean() + d[-1,:,zf].mean() + d[:,0,zf].mean() + d[:,-1,zf].mean())/4
    
    #sum over all pixels (and hence mean) should be preserved over z (for widefield psf)
    dm = d.mean(1).mean(0)
    
    bg = dm - dm[zf]
    
    return np.maximum(d - bg[None, None, :], 0) +  1e-5
