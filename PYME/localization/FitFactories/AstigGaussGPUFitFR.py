#!/usr/bin/python

##################
# AstigGaussGPUFitFR.py
#
# This FitFactory calls the pyme-warp-drive package to perform
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


from PYME.Analysis._fithelpers import *
from PYME.localization.FitFactories.fitCommon import pack_results
import logging
logger = logging.getLogger(__name__)

def astigmatic_gaussian(p, X, Y):
    """
    2D Gaussian model function with independent sigx, sigy

    Parameters
    ----------
    p: iterable
        Parameter array, in the following order:
            x0: float
                x center position [units same as X, Y]
            y0: float
                y center position [units same as X, Y]
            A: float
                Amplitude, or peak height of the Gaussian [units same as b]
            b: float
                background, constant offset [units same as A]
            sx: float
                sigma along the x direction [units same as X, Y]
            sy: float
                sigma along the y direction [units same as X, Y]
    X: ndarray
        y position array (1d)
    Y: ndarray
        y position array (1d)

    Returns
    -------
    model: ndarray
        2D image of gaussian

    Notes
    -----
    This function is not actually used by the GPU fit, but is here as an example, and for testing with, e.g.
    PYME.localization.Test.fitTestJigSCMOS. See fitTestJig notes about units, particularly the amplitude parameters A
    and b.
    """
    x0, y0, A, b, sx, sy = p
    return A * np.exp(-(X[:,None]-x0)**2/(2*sx**2) - (Y[None,:] - y0)**2/(2*sy**2)) + b

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

_warpdrive = None

missing_warpdrive_msg = """
Could not import the warpdrive module. GPU fitting requires the warpdrive module,
which is distributed separately due to licensing issues. The warpdrive module is
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
            Uncorrected data, in raw ADU
        metadata : PYME.IO.MetaDataHandler.MDHandlerBase or derived class
        background : numpy.ndarray, warpdrive.buffers.Buffer, or scalar
            warpdrive.buffers.Buffer allows asynchronous estimation of the per-pixel background on the GPU.
        """

        self.data = data
        self.metadata = metadata

        if isinstance(background, np.ndarray):
            self.background = background.squeeze()  # will be set to contiguous float32 inside of detector class method
        elif np.isscalar(background):
            self.background = np.ascontiguousarray(background * np.ones_like(self.data), dtype=np.float32)
        else:  # it's a buffer!
            self.background = background


    def refresh_warpdrive(self, cameraMaps):
        try:
            import warpdrive
        except ImportError:
            print("GPU fitting available on-request for academic use. Please contact David Baddeley or Joerg Bewersdorf.")

            raise ImportError(missing_warpdrive_msg)

        global _warpdrive  # One instance for each process, re-used for subsequent fits.

        # get darkmap [ADU]
        self.darkmap = cameraMaps.getDarkMap(self.metadata)
        if np.isscalar(self.darkmap):
            self.darkmap = self.darkmap * np.ones_like(self.data)

        # get flatmap [unitless]
        self.flatmap = cameraMaps.getFlatfieldMap(self.metadata)
        if np.isscalar(self.flatmap):
            self.flatmap = self.flatmap * np.ones_like(self.data)

        # get varmap [e-^2]
        self.varmap = cameraMaps.getVarianceMap(self.metadata)
        if np.isscalar(self.varmap):
            if self.varmap == 0:
                self.varmap = np.ones_like(self.data)
                logger.error('Variance map not found and read noise defaulted to 0; changing to 1 to avoid x/0.')
            self.varmap = self.varmap*np.ones_like(self.data)

        if isinstance(self.background, np.ndarray):  # flatfielding is done on CPU-calculated backgrounds
            # fixme - do we change this control flow in remfitbuf by doing our own sigma calc?
            self.background = self.background/self.flatmap  # no unit conversion here, still in [ADU]
        else:
            # if self.background is a buffer, the background is already on the GPU and has not been flatfielded
            pass

        # Account for any changes we need to make to the detector instance
        small_filter_size = self.metadata.getOrDefault('Analysis.DetectionFilterSize', 
                                                       4)
        guess_psf_sigma_pix = self.metadata.getOrDefault('Analysis.GuessPSFSigmaPix',
                                                         600 / 2.8 / (self.metadata.voxelsize_nm.x))

        if not _warpdrive:  # if we don't have a detector, make one and return
            _warpdrive = warpdrive.detector(small_filter_size, 2 * small_filter_size, guess_psf_sigma_pix)
            _warpdrive.allocate_memory(np.shape(self.data))
            _warpdrive.prepare_maps(self.darkmap, self.varmap, self.flatmap, self.metadata['Camera.ElectronsPerCount'],
                                    self.metadata['Camera.NoiseFactor'], self.metadata['Camera.TrueEMGain'])
            return

        need_maps_filtered, need_mem_allocated = False, False
        # check if our filter sizes are the same
        if small_filter_size != _warpdrive.small_filter_size:
            _warpdrive.set_filter_kernels(small_filter_size, 2 * small_filter_size)
            # need to rerun map filters
            need_maps_filtered = True

        # check if the data is coming from a different camera region
        if _warpdrive.varmap.shape != self.varmap.shape:
            need_mem_allocated, need_maps_filtered = True, True
        else:
            # check if both corners are the same
            top_left = np.array_equal(self.varmap[:20, :20], _warpdrive.varmap[:20, :20])
            bot_right = np.array_equal(self.varmap[-20:, -20:], _warpdrive.varmap[-20:, -20:])
            if not (top_left or bot_right):
                need_maps_filtered = True

        if need_mem_allocated:
            _warpdrive.allocate_memory(np.shape(self.data))
        if need_maps_filtered:
            _warpdrive.prepare_maps(self.darkmap, self.varmap, self.flatmap,
                                    self.metadata['Camera.ElectronsPerCount'], self.metadata['Camera.NoiseFactor'],
                                    self.metadata['Camera.TrueEMGain'])

    def get_results(self):
        # LLH: (N); dpars and CRLB (N, 6)
        #convert pixels to nm; voxelsize in units of um
        voxelsize=self.metadata.voxelsize_nm
        
        dpars = np.reshape(_warpdrive.fit_res, (_warpdrive.n_max_candidates_per_frame, 6))[:_warpdrive.n_candidates, :]
        dpars[:, 0] *= (voxelsize.y)
        dpars[:, 1] *= (voxelsize.x)
        dpars[:, 4] *= (voxelsize.y)
        dpars[:, 5] *= (voxelsize.x)

        LLH = None
        if _warpdrive.calculate_crb:
            LLH = _warpdrive.LLH[:_warpdrive.n_candidates]
            CRLB = np.reshape(_warpdrive.CRLB, (_warpdrive.n_max_candidates_per_frame, 6))[:_warpdrive.n_candidates, :]
            # fixme: Should never have negative CRLB, yet Yu reports ocassional instances in Matlab verison, check
            CRLB[:, 0] = np.sqrt(np.abs(CRLB[:, 0]))*(voxelsize.y)
            CRLB[:, 1] = np.sqrt(np.abs(CRLB[:, 1]))*(voxelsize.x)
            CRLB[:, 4] = np.sqrt(np.abs(CRLB[:, 4]))*(voxelsize.y)
            CRLB[:, 5] = np.sqrt(np.abs(CRLB[:, 5]))*(voxelsize.x)

        #return self.chan1.dpars # each fit produces column vector of results, append them all horizontally for return
        res_list = np.empty(_warpdrive.n_candidates, FitResultsDType)
        resultCode = 0
        
        tIndex = int(self.metadata.getOrDefault('tIndex', 0))

        # package our results with the right labels
        if _warpdrive.calculate_crb:
            for ii in range(_warpdrive.n_candidates):
                res_list[ii] = pack_results(fresultdtype, tIndex=tIndex, fitResults=dpars[ii, :], fitError=CRLB[ii, :],
                                           LLH=LLH[ii], resultCode=resultCode, nFit=_warpdrive.n_candidates)
        else:
            for ii in range(_warpdrive.n_candidates):
                res_list[ii] = pack_results(fresultdtype, tIndex=tIndex, fitResults=dpars[ii, :], fitError=None,
                                           LLH=LLH[ii], resultCode=resultCode, nFit=_warpdrive.n_candidates)

        return np.hstack(res_list)

    def FindAndFit(self, threshold, cameraMaps, **kwargs):
        """

        Parameters
        ----------
        threshold: float
            detection threshold, as a multiple of per-pixel noise standard deviation
        cameraMaps: cameraInfoManager object (see remFitBuf.py)

        Returns
        -------
        results

        """
        # make sure we've loaded and pre-filtered maps for the correct FOV
        self.refresh_warpdrive(cameraMaps)

        # prepare the frame
        _warpdrive.prepare_frame(self.data)

        #PYME ROISize is a half size
        roi_size = int(2*self.metadata.getEntry('Analysis.ROISize') + 1)

        ########## Actually do the fits #############
        # toggle background subtraction (done both in detection and the fit)
        if self.metadata.getOrDefault('Analysis.subtractBackground', False):
            _warpdrive.difference_of_gaussian_filter(self.background)
        else:
            _warpdrive.difference_of_gaussian_filter()

        _warpdrive.get_candidates(threshold, roi_size)

        if _warpdrive.n_candidates == 0:  # exit if our job is already done
            resList = np.empty(0, FitResultsDType)
            return resList
        _warpdrive.fit_candidates(roi_size)
        return self.get_results()

    def FromPoint(self, x, y, roiHalfSize=None):
        """
        This is a bit hacked to work with PYME.localization.Test.fitTestJigSCMOS
        """
        # fixme - surely just broke units on this
        from PYME.localization import remFitBuf
        camera_maps = remFitBuf.CameraInfoManager()

        self.refresh_warpdrive(camera_maps)
        roi_size = int(2*self.metadata.getOrDefault('Analysis.ROISize', 7.5) + 1)
        # make sure our ROI size is smaller than the data, otherwise we'll have some bad memory accesses
        roi_size = min(roi_size, self.data.shape[0] - 3)
        _warpdrive.fitFunc = _warpdrive.gaussAstig
        # todo - add check on x and y to make sure we don't have bad memory accesses for small self.data
        _warpdrive.insertTestCandidates([int(x) + int(y)*self.data.shape[0]])
        _warpdrive.insertData(self.data, self.background)
        _warpdrive.fit_candidates(roi_size)
        return self.get_results()


    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*np.mgrid[(y - roiHalfSize):(y + roiHalfSize + 1)]

        return (astigmatic_gaussian(params, X, Y), X[0], Y[0], 0)

# so that fit tasks know which class to use
FitFactory = GaussianFitFactory
#FitResult = GaussianFitResultR
FitResultsDType = fresultdtype  #only defined if returning data as numarray


GPU_PREFIT = True  # fit factory does its own prefit steps on the GPU

import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.FloatParam('Analysis.ROISize', u'ROI half size', 7.5),
    mde.BoolParam('Analysis.GPUPCTBackground', 'Calculate percentile background on GPU', True),
    mde.IntParam('Analysis.DetectionFilterSize', 'Detection Filter Size:', 4,
                 'Filter size used for point detection; units of pixels. Should be slightly less than the PSF FWHM'),
]

DESCRIPTION = 'Astigmatic Gaussian fitting performed at warp-speed on the GPU'
LONG_DESCRIPTION = 'Astigmatic Gaussian fitting on the GPU: Fits astigmatic gaussian with sCMOS noise model. Uses it\'s own object detection routine'
USE_FOR = '3D via Astigmatism, with sCMOS noise model'
