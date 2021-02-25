# Copyright David Baddeley, Andrew Barentine, 2017
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
from .fitCommon import fmtSlicesUsed, pack_results
from . import FFBase
from PYME.Analysis._fithelpers import FitModelWeighted
from scipy.signal import convolve2d


##--------------------------------------------------------------------------------------------------------------------##
#  Model functions

# define factor by which to interpolate grid for evaluating numerical convolutions. Note the data is never interpolated
INTERP_FACTOR = 5

def bead(X, Y, beadDiam):
    """
    returns a sum-normalized bead projection
    """
    x0, y0 = np.mean(X), np.mean(Y)
    beadRad = 0.5*beadDiam
    beadProj = np.nan_to_num(np.sqrt(beadRad**2 - (X[:,None] - x0)**2 - (Y[None, :] - y0)**2))
    # return normalized bead projection
    return (1/np.sum(beadProj))*beadProj

def f_gaussAstigBead(p, X, Y, beadDiam):
    """
    astigmatic Gaussian model function with linear background (parameter vector [A, x0, y0, sx, sy, b, b_x, b_y])
    convolved with a bead shape
    """
    A, x0, y0, sx, sy, c, b_x, b_y = p

    # interpolate, as the bead diameter will usually be roughly the size of a pixel
    step = (X[1]-X[0])/float(INTERP_FACTOR)  # NB this assumes X and Y spacing is equal, i.e. square pixels
    xx = np.arange(X[0] - INTERP_FACTOR*step, X[-1] + (INTERP_FACTOR + 1)*step, step)
    yy = np.arange(Y[0] - INTERP_FACTOR*step, Y[-1] + (INTERP_FACTOR + 1)*step, step)

    gauss = A * np.exp(-(xx[:, None] - x0) ** 2 / (2 * sx ** 2) - (yy[None, :] - y0) ** 2 / (2 * sy ** 2)) + c + b_x * xx[:,None] + b_y * yy[None,:]
    # how many steps to include beadDiam
    ns = np.ceil(beadDiam/step).astype(int) + 1
    beadProj = bead(xx[:2*ns], yy[:2*ns], beadDiam)

    model = convolve2d(gauss, beadProj, mode='same')

    return model[INTERP_FACTOR:-INTERP_FACTOR:INTERP_FACTOR, INTERP_FACTOR:-INTERP_FACTOR:INTERP_FACTOR]

def f_rotGaussAstigBead(p, X, Y, beadDiam, theta = 0):
    """
    astigmatic Gaussian model function with linear background (parameter vector [A, x0, y0, sx, sy, b, b_x, b_y])
    rotated by theta radians and convolved with a bead shape.
    """
    A, x0, y0, sx, sy, c, b_x, b_y = p

    # interpolate, as the bead diameter will usually be roughly the size of a pixel
    step = (X[1]-X[0])/float(INTERP_FACTOR)  # NB this assumes X and Y spacing is equal, i.e. square pixels
    xx = np.arange(X[0] - INTERP_FACTOR*step, X[-1] + (INTERP_FACTOR + 1)*step, step)
    yy = np.arange(Y[0] - INTERP_FACTOR*step, Y[-1] + (INTERP_FACTOR + 1)*step, step)

    #gauss = A * np.exp(-(xx[:, None] - x0) ** 2 / (2 * sx ** 2) - (yy[None, :] - y0) ** 2 / (2 * sy ** 2)) + c + b_x * xx[:,None] + b_y * yy[None,:]
    alpha = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
    beta = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
    charlie = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
    rotGauss = A * np.exp(-(alpha*(xx[:, None] - x0)**2 - 2*beta*(xx[:, None] - x0)*(yy[None, :] - y0) + charlie*(yy[None, :] - y0)**2))
    # how many steps to include beadDiam
    ns = np.ceil(beadDiam/step) + 1
    beadProj = bead(xx[:2*ns], yy[:2*ns], beadDiam)

    model = convolve2d(rotGauss, beadProj, mode='same')

    return model[INTERP_FACTOR:-INTERP_FACTOR:INTERP_FACTOR, INTERP_FACTOR:-INTERP_FACTOR:INTERP_FACTOR]





##--------------------------------------------------------------------------------------------------------------------##
# define the data type we're going to return
fresultdtype = [('tIndex', '<i4'),
                ('fitResults', [('A', '<f4'),
                                ('x0', '<f4'), ('y0', '<f4'),
                                ('sigmax', '<f4'),
                                ('sigmay', '<f4'),
                                ('background', '<f4'),
                                ('bx', '<f4'),
                                ('by', '<f4')]),
                ('fitError', [('A', '<f4'),
                              ('x0', '<f4'),
                              ('y0', '<f4'),
                              ('sigmax', '<f4'),
                              ('sigmay', '<f4'),
                              ('background', '<f4'),
                              ('bx', '<f4'),
                              ('by', '<f4')]),
                ('startParams', [('A', '<f4'),
                                 ('x0', '<f4'),
                                 ('y0', '<f4'),
                                 ('sigmax', '<f4'),
                                 ('sigmay', '<f4'),
                                 ('background', '<f4'),
                                 ('bx', '<f4'),
                                 ('by', '<f4')]),
                ('resultCode', '<i4'),
                ('slicesUsed', [('x', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                                ('y', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                                ('z', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')])]),
                ('subtractedBackground', '<f4'),
                ('meanSquaredError', '<f4')
                ]


class GaussianFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_gaussAstigBead, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma, **kwargs)

        self.solver = FitModelWeighted
        try:  # check if bead diameter is stored in the metadata
            self.beadDiam = metadata['Bead.Diameter']  # Should be in [nm]
        except KeyError:
            try:  # check if a bead diameter has been injected in just for analysis
                self.beadDiam = metadata['Analysis.Bead.Diameter']
            except KeyError:
                raise UserWarning('BeadConvolvedAstigGauss fitfactory requires a Bead.Diameter entry in the metadata')

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background

        # estimate some start parameters...
        A = data.max() - data.min()  # amplitude

        vs = self.metadata.voxelsize_nm
        x0 = vs.x * x
        y0 = vs.y * y

        bgm = np.mean(background)

        startParameters = [A, x0, y0, 250 / 2.35, 250 / 2.35, dataMean.min(), .001, .001]

        # do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y, self.beadDiam)

        # try to estimate errors based on the covariance matrix
        fitErrors = None
        try:
            fitErrors = np.sqrt(
                np.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataMean.ravel()) - len(res)))
            if np.any(np.isnan(fitErrors)):
                # for some reason we occasionally get negatives on the diagonal of the covariance matrix (and NaN for the fitError.
                # this shouldn't happen, but catch it here in case and flag the fit as having failed
                fitErrors = None
        except Exception:
            pass
        
        tIndex = int(self.metadata.getOrDefault('tIndex', 0))
        return pack_results(fresultdtype, tIndex=tIndex, fitResults=res, 
                            fitError=fitErrors, startParams=np.array(startParameters), 
                            resultCode=resCode, slicesUsed=(xslice, yslice, zslice),
                            subtractedBackground=bgm, 
                            meanSquaredError=np.mean(infodict['fvec']**2))

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        """Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation"""
        # generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x * np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y * np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return 'FIXME' #  (f_gaussAstigBead(params, X, Y, beadDiam), X[0], Y[0], 0)


# so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResultsDType = fresultdtype  # only defined if returning data as numarray

import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.IntParam('Analysis.ROISize', u'ROI half size', 7),

]

DESCRIPTION = 'Fitting astigmatic gaussian after convolution with bead shape'
LONG_DESCRIPTION = 'Fitting astigmatic gaussian after convolution with bead shape'
USE_FOR = '3D astigmatism calibrations'
