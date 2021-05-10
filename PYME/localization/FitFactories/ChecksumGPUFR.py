import scipy.ndimage as ndimage
import numpy as np
from .fitCommon import fmtSlicesUsed, pack_results

from PYME.localization.cModels.gauss_app import *
from PYME.Analysis._fithelpers import *


fresultdtype=[
    ('tIndex', '<i4'), ('ch_data_cpu', '<u4'), ('ch_data_e_gpu', '<u4'),
    ('ch_sigma', '<u4'), ('ch_background', '<u4'), ('ch_dog', '<u4'),
    ('ch_max_filter', '<u4'),
]

_warpdrive = None

missing_warpdrive_msg = """
Could not import the warpdrive module. GPU fitting requires the warpdrive module,
which is distributed separately due to licensing issues. The warpdrive module is
available on request and is free for academic use. Please contact David Baddeley
or Joerg Bewersdorf.
"""

class ChecksumFitFactory(object):
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
        import zlib
        try:
            import warpdrive
        except ImportError:
            print("GPU fitting available on-request for academic use. Please contact David Baddeley or Joerg Bewersdorf.")
            raise ImportError(missing_warpdrive_msg)

        res = np.zeros(1, fresultdtype)
        res['tIndex'] = self.metadata.tIndex
        res['ch_data_cpu'] = zlib.crc32(self.data)
        # make sure we've loaded and pre-filtered maps for the correct FOV
        self.refresh_warpdrive(cameraMaps)
        # prepare the frame
        _warpdrive.prepare_frame(self.data)
        warpdrive.cuda.memcpy_dtoh_async(_warpdrive.dtarget, _warpdrive.data_gpu,
                                         stream=_warpdrive.main_stream_r)
        _warpdrive.main_stream_r.synchronize()
        res['ch_data_e_gpu'] = zlib.crc32(_warpdrive.dtarget)
        warpdrive.cuda.memcpy_dtoh_async(_warpdrive.dtarget, _warpdrive.noise_sigma_gpu,
                                         stream=_warpdrive.main_stream_r)
        _warpdrive.main_stream_r.synchronize()
        res['ch_sigma'] = zlib.crc32(_warpdrive.dtarget)
        

        #PYME ROISize is a half size
        roi_size = int(2*self.metadata.getEntry('Analysis.ROISize') + 1)

        ########## Actually do the fits #############
        # toggle background subtraction (done both in detection and the fit)
        if self.metadata.getOrDefault('Analysis.subtractBackground', False):
            _warpdrive.difference_of_gaussian_filter(self.background)
        else:
            _warpdrive.difference_of_gaussian_filter()
        
        warpdrive.cuda.memcpy_dtoh_async(_warpdrive.dtarget, _warpdrive.bkgnd_gpu,
                                         stream=_warpdrive.main_stream_r)
        _warpdrive.main_stream_r.synchronize()
        res['ch_background'] = zlib.crc32(_warpdrive.dtarget)
        warpdrive.cuda.memcpy_dtoh_async(_warpdrive.dtarget, _warpdrive.unif1_gpu,
                                         stream=_warpdrive.main_stream_r)
        _warpdrive.main_stream_r.synchronize()
        res['ch_dog'] = zlib.crc32(_warpdrive.dtarget)

        _warpdrive.get_candidates(threshold, roi_size)
        warpdrive.cuda.memcpy_dtoh_async(_warpdrive.dtarget, _warpdrive.maxf_data_gpu,
                                         stream=_warpdrive.main_stream_r)
        _warpdrive.main_stream_r.synchronize()
        res['ch_max_filter'] = zlib.crc32(_warpdrive.dtarget)

        return res


   
        


GPU_PREFIT = True  # fit factory does its own prefit steps on the GPU

import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.FloatParam('Analysis.ROISize', u'ROI half size', 7.5),
    mde.BoolParam('Analysis.GPUPCTBackground', 'Calculate percentile background on GPU', True),
    mde.IntParam('Analysis.DetectionFilterSize', 'Detection Filter Size:', 4,
                 'Filter size used for point detection; units of pixels. Should be slightly less than the PSF FWHM'),
]
#so that fit tasks know which class to use
FitFactory = ChecksumFitFactory

FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = 'CRC32 checksum of frame'
LONG_DESCRIPTION = 'Takes a CRC32 checksum of frame'
USE_FOR = 'Debugging to ensure frame data and background are reproducible'
