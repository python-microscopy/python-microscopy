#!/usr/bin/python

##################
# AstigGaussGPUFitFR.py
#
# This FitFactory calls the PYMEwarpDrive package to perform
# candidate molecule detection and fitting on the GPU, which
# takes into account an sCMOS specific noise model. This
# GPU package is available on-request for academic use. If
# interested, please contact David Baddeley or Joerg Bewersdorf.
# The fitting itself is described in DOI: 10.1038/nmeth.2488
#
# Copyright David Baddeley, Andrew Barentine 2016
# david.baddeley@yale.edu
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
# AESB 04,2016: modified GaussMultiFitSR to work with PYMEwarpDrive
##################


from PYME.Analysis._fithelpers import *
import logging
logger = logging.getLogger(__name__)

##################
# Model Function, only for reference in this case.
def f_gaussAstigSlow(p, X, Y):
    """2D Gaussian model function with independent sigx, sigy - parameter vector [A, x0, y0, sx, sy]"""
    #x0, y0, A, sx, sy, c, = p
    x0, y0, A, sx, sy = p
    return A*np.exp(-(X[:,None]-x0)**2/(2*sy**2) - (Y[None,:] - y0)**2/(2*sx**2))  # + c
#####################


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('y0', '<f4'), ('x0', '<f4'), #Fang and Davids xys are swapped
                              ('A', '<f4'),
                              ('background', '<f4'),
                              ('sigmay', '<f4'),('sigmax', '<f4')]),

              ('fitError', [('y0', '<f4'), ('x0', '<f4'),
                              ('A', '<f4'),
                              ('background', '<f4'),
                              ('sigmay', '<f4'),('sigmax', '<f4')]),
              ('resultCode', '<i4'),
              ('LLH', '<f4'),
              ('nFit', '<i4')]


def GaussianFitResultR(fitResults, metadata, resultCode=-1, fitErr=None, LLH=None, nEvents=1):
    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')
        LLH = np.zeros(nEvents)

    tIndex = metadata.getOrDefault('tIndex', 0)

    return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, LLH, nEvents)], dtype=fresultdtype)

_warpDrive = None

missing_warpDrive_msg = """
Could not import the warpDrive module. GPU fitting requires the warpDrive module,
which is distributed separately due to licensing issues. The warpDrive module is
available on request and is free for academic use. Please contact David Baddeley
or Joerg Bewersdorf.
"""

class GaussianFitFactory:
    X = None
    Y = None

    def __init__(self, data, metadata, fitfcn=None, background=None, noiseSigma=None, **kwargs):
        """

        Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata.

        Parameters
        ----------
        data : numpy.ndarray
        metadata : PYME.IO.MetaDataHandler.MDHandlerBase or derived class
        fitfcn : dummy variable
            Not used in this fit factory
        background : numpy.ndarray or warpDrive.buffers.Buffer
            warpDrive.buffers.Buffer allows asynchronous estimation of the per-pixel background on the GPU.
        noiseSigma : numpy.ndarray
            (over-)estimate of the noise level at each pixel (see fitTask.calcSigma in remFitBuf.py)
        """

        self.data = np.squeeze(data).astype(np.float32)  # np.ascontiguousarray(np.squeeze(data), dtype=np.float32)
        self.metadata = metadata

        if isinstance(background, np.ndarray):
            self.background = background.squeeze()  # will be set to contiguous float32 inside of detector class method
        else:  # it's a buffer!
            self.background = background
        self.noiseSigma = noiseSigma
        self.fitfcn = fitfcn


    def refreshWarpDrive(self, cameraMaps):
        try:
            import warpDrive
        except ImportError:
            print("GPU fitting available on-request for academic use. Please contact David Baddeley or Joerg Bewersdorf.")

            raise ImportError(missing_warpDrive_msg)

        global _warpDrive  # One warpDrive instance for each process, re-used for subsequent fits.

        # get varmap and flatmap
        varmap = cameraMaps.getVarianceMap(self.metadata)
        if not np.isscalar(varmap):
            self.varmap = varmap.astype(np.float32)  # np.ascontiguousarray(varmap)
        else:
            if varmap == 0:
                self.varmap = np.ones_like(self.data)
                logger.error('Variance map not found and read noise defaulted to 0; changing to 1 to avoid x/0.')
            else:
                self.varmap = varmap*np.ones_like(self.data)

        flatmap = cameraMaps.getFlatfieldMap(self.metadata)
        if not np.isscalar(flatmap):
            self.flatmap = flatmap.astype(np.float32)  # flatmat is mean-normalized and unitless
        else:
            #flatmap = self.metadata['Camera.TrueEMGain']*self.metadata['Camera.ElectronsPerCount']
            self.flatmap = flatmap*np.ones_like(self.data) # flatmat is mean-normalized and unitless

        # subtract darkmap
        #### DB - Dark map is already subtracted by remFitBuf!!!! NOTE: flatfielding will also have been done, so undo
        #darkmap = cameraMaps.getDarkMap(self.metadata)
        #self.data -= darkmap #same op for scalar or array

        ### Undo the flatfielding we did in remFitBuf: (img.astype('f')-dk)*flat
        self.data = self.data/self.flatmap  # no conversion here, flatmap normed so data still in [ADU]
        # if self.background is not None:  # flatfielding is also done on moving-averaged background
        if isinstance(self.background, np.ndarray):
            self.background = self.background/self.flatmap  # no conversion here, flatmap normed so data still in [ADU]
        else:
            # if self.background is a buffer, the background is already on the GPU and has not been flatfielded
            pass

        # Account for any changes we need to make in memory allocation on the GPU
        if not _warpDrive:  # Initialize new detector object for this process
            guess_psf_sigma_pix = self.metadata.getOrDefault('Analysis.GuessPSFSigmaPix',
                                                             600 / 2.8 / (self.metadata['voxelsize.x'] * 1e3))
            small_filter_size = self.metadata.getEntry('Detection.FilterSize')
            large_filter_size = 2 * small_filter_size
            _warpDrive = warpDrive.detector(small_filter_size, large_filter_size, guess_psf_sigma_pix)
            _warpDrive.allocateMem(np.shape(self.data), self.data.dtype.itemsize)
            _warpDrive.prepvar(self.varmap, self.flatmap, self.metadata['Camera.ElectronsPerCount'])

            #If the data is coming from a different region of the camera, reallocate
        elif _warpDrive.data.shape == self.data.shape:
            # check if both corners are the same
            topLeft = np.array_equal(self.varmap[:20, :20], _warpDrive.varmap[:20, :20])
            botRight = np.array_equal(self.varmap[-20:, -20:], _warpDrive.varmap[-20:, -20:])
            if not (topLeft or botRight):
                _warpDrive.prepvar(self.varmap, self.flatmap, self.metadata['Camera.ElectronsPerCount'])
        else:  # data is a different shape - we know that we need to re-allocate and prepvar
            _warpDrive.allocateMem(np.shape(self.data), self.data.dtype.itemsize)
            _warpDrive.prepvar(self.varmap, self.flatmap, self.metadata['Camera.ElectronsPerCount'])

    def getRes(self):
        # LLH: (N); dpars and CRLB (N, 6)
        #convert pixels to nm; voxelsize in units of um
        dpars = np.reshape(_warpDrive.dpars, (_warpDrive.maxCandCount, 6))[:_warpDrive.candCount, :]
        dpars[:, 0] *= (1000*self.metadata.voxelsize.y)
        dpars[:, 1] *= (1000*self.metadata.voxelsize.x)
        dpars[:, 4] *= (1000*self.metadata.voxelsize.y)
        dpars[:, 5] *= (1000*self.metadata.voxelsize.x)

        fitErrors=None
        LLH = None


        if _warpDrive.calcCRLB:
            LLH = _warpDrive.LLH[:_warpDrive.candCount]
            CRLB = np.reshape(_warpDrive.CRLB, (_warpDrive.maxCandCount, 6))[:_warpDrive.candCount, :]
            # fixme: Should never have negative CRLB, yet Yu reports ocassional instances in Matlab verison, check
            CRLB[:, 0] = np.sqrt(np.abs(CRLB[:, 0]))*(1000*self.metadata.voxelsize.y)
            CRLB[:, 1] = np.sqrt(np.abs(CRLB[:, 1]))*(1000*self.metadata.voxelsize.x)
            CRLB[:, 4] = np.sqrt(np.abs(CRLB[:, 4]))*(1000*self.metadata.voxelsize.y)
            CRLB[:, 5] = np.sqrt(np.abs(CRLB[:, 5]))*(1000*self.metadata.voxelsize.x)

        #return self.chan1.dpars # each fit produces column vector of results, append them all horizontally for return
        resList = np.empty(_warpDrive.candCount, FitResultsDType)
        resultCode = 0

        # package our results with the right labels
        for ii in range(_warpDrive.candCount):
            resList[ii] = GaussianFitResultR(dpars[ii, :], self.metadata, resultCode, CRLB[ii, :], LLH[ii], _warpDrive.candCount)

        return np.hstack(resList)

    def FindAndFit(self, threshold=4, gui=False, cameraMaps=None):
        """

        Args:
            threshold: in units of noiseSigma (if supplied)
            gui: unused
            cameraMaps: cameraInfoManager object (see remFitBuf.py)
            noiseSigma: (over-)estimate of the noise level at each pixel (see fitTask.calcSigma in remFitBuf.py)

        Returns:
            output of self.getRes

        """
        # make sure we've loaded and pre-filtered maps for the correct FOV
        self.refreshWarpDrive(cameraMaps)

        # use signal to noise thresholding if available
        #thresh = threshold*noiseSigma if (noiseSigma is not None) else threshold

        #PYME ROISize is a half size
        roiSize = int(2*self.metadata.getEntry('Analysis.ROISize') + 1)

        ########## Actually do the fits #############
        # toggle background subtraction (done both in detection and the fit)
        if self.metadata.getOrDefault('Analysis.subtractBackground', False):
            _warpDrive.smoothFrame(self.data, self.background)
        else:
            _warpDrive.smoothFrame(self.data)
        _warpDrive.getCand(threshold, roiSize, self.noiseSigma, self.metadata['Camera.ElectronsPerCount'])
        if _warpDrive.candCount == 0:
            resList = np.empty(0, FitResultsDType)
            return resList
        _warpDrive.fitItToWinIt(roiSize)
        return self.getRes()

    def FromPoint(self, x, y, roiHalfSize=None):
        # FIXME - currently ignoring roiHalfSize from PYME because it is typically an int
        from PYME.localization import remFitBuf
        cameraMaps = remFitBuf.CameraInfoManager()

        self.refreshWarpDrive(cameraMaps)

        roiSize = int(2*self.metadata.getOrDefault('Analysis.ROISize', 7.5) + 1)

        _warpDrive.fitFunc = _warpDrive.gaussAstig
        _warpDrive.insertTestCandidates(int(x) + int(y)*_warpDrive.rsize)  # 1799)
        _warpDrive.insertData(self.data)
        _warpDrive.fitItToWinIt(roiSize)

        return self.getRes()


    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        X = 1e3*md.voxelsize.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = 1e3*md.voxelsize.y*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_gaussAstigSlow(params, X, Y), X[0], Y[0], 0)

# so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype  #only defined if returning data as numarray

#this means that factory is responsible for it's own object finding and implements
#a GetAllResults method that returns a list of localisations
MULTIFIT=True
# GPU_BUFFER_READY means this factory can accept a GPU buffer object to reduce data transfers.
# TODO - differ correctImage(data) call from remFitBuf to GPU fit factory with this same flag
GPU_BUFFER_READY=True

import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.FloatParam('Analysis.ROISize', u'ROI half size', 7.5),
    mde.BoolParam('Analysis.GPUPCTBackground', 'Calculate percentile background on GPU', True),
    mde.IntParam('Detection.FilterSize', 'Detection Filter Size:', 4,
                 'Filter size used for point detection; units of pixels. Should be slightly less than the PSF FWHM'),
]

DESCRIPTION = 'Astigmatic Gaussian fitting performed at warp-speed on the GPU'
LONG_DESCRIPTION = 'Astigmatic Gaussian fitting on the GPU: Fits astigmatic gaussian with sCMOS noise model. Uses it\'s own object detection routine'
USE_FOR = '3D via Astigmatism, with sCMOS noise model'
