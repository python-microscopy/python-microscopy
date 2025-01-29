# -*- coding: utf-8 -*-

"""
Created on Mon August 10 2020

@author: zacsimile
"""

from PYME.Acquire.Hardware.Camera import Camera
from PYME.Acquire import eventLog

import numpy as np
import logging
logger = logging.getLogger(__name__)

import ctypes
import platform
import time

sys = platform.system()
if sys != 'Windows':
    raise Exception("Operating system is not supported.")

# We use https://pypi.org/project/pco/
# For API docs, see https://www.pco.de/fileadmin/user_upload/pco-manuals/pco.sdk_manual.pdf and
# https://www.pco.de/fileadmin/user_upload/pco-manuals/pco.recorder_manual.pdf
try:
    import pco
except ModuleNotFoundError:
    raise ModuleNotFoundError('Please install the pco package from https://pypi.org/project/pco/.')

timebase = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3}  # Conversions from a pco dictionary

class PcoCam(Camera):
    numpy_frames = 1
    
    def __init__(self, camNum, debuglevel='off'):
        Camera.__init__(self)
        self.camNum = camNum
        self.initalized = False
        self.noiseProps = None
        self._debuglevel = debuglevel  # ['off', 'error', 'verbose', 'extra verbose']

    def Init(self):
        self.cam = pco.Camera(debuglevel=self._debuglevel)
        self.SetDescription()
        self._mode = self.MODE_CONTINUOUS
        self.SetHotPixelCorrectionMode('off')
        self.buffer_size = 0
        self.n_read = 0
        self.initalized = True
        self._roi = None
        self.SetROI(1, 1, self.GetCCDWidth(), self.GetCCDHeight())
        self._ccd_temp = 0 
        self._ccd_temp_set_point = 0
        self._electr_temp = 0
        self._cycle_time = 0
        self._integ_time = 0
        self._binning_x = 0
        self._binning_y = 0
        self.recording = False

    @property
    def noise_properties(self):
        return self.noiseProps

    def SetDescription(self):
        self.desc = self.cam.sdk.get_camera_description()

    def ExpReady(self):
        return self.recording and (self.GetNumImsBuffered() >= 1)

    def GetName(self):
        return self.cam.sdk.get_camera_name()['camera name']

    def CamReady(self):
        # return self.cam.sdk.get_camera_busy_status()['busy status'] == 0
        return self.initalized

    def ExtractColor(self, chSlice, mode):
        # Somehow this check matters... shouldn't this be taken care of by ExpReady???
        if (not self.recording) or (self.GetNumImsBuffered() < 1):
            raise RuntimeError('Trying to grab a frame when there is no frame to grab.')

        if self.n_read < self.buffer_size:
            curr_frame = self.n_read
        else:
            curr_frame = self.buffer_size-self.GetNumImsBuffered()

        # Grab the image (unused metadata _)
        image, _ = self.cam.image(curr_frame)

        ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                   image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), 
                   chSlice.nbytes)

        self.n_read += 1

    def GetHeadModel(self):
        return self.cam.sdk.get_camera_type()['camera type']

    def GetSerialNumber(self):
        return str(self.cam.sdk.get_camera_type()['serial number'])

    def SetIntegTime(self, time):
        lb = self.desc['Min Expos DESC']*1e-9  # ns
        ub = self.desc['Max Expos DESC']*1e-3  # ms
        step = self.desc['Min Expos Step DESC']*1e-9  # ns
        
        # Round to a multiple of time step
        time = np.floor(time/step)*step

        # Don't let this go out of bounds
        time = np.clip(time, lb, ub)
            
        self.cam.set_exposure_time(time)
        d = self.cam.sdk.get_delay_exposure_time()
        self._integ_time = d['exposure']*timebase[d['exposure timebase']] 

    def GetIntegTime(self):
        return self._integ_time

    def GetCycleTime(self):
        return self._cycle_time

    def GetCCDWidth(self):
        return int(self.desc['max. horizontal resolution standard'])

    def GetCCDHeight(self):
        return int(self.desc['max. vertical resolution standard'])

    def GetPicWidth(self):
        return self.GetROI()[2] - self.GetROI()[0] + 1

    def GetPicHeight(self):
        return self.GetROI()[3] - self.GetROI()[1] + 1

    def SetHorizontalBin(self, value):
        b_max, step_type = self.desc['max. binning horizontal'], self.desc['binning horizontal stepping']

        if step_type == 0:
            # binary binning, round to powers of two
            value = 1<<(value-1).bit_length()
        else:
            value = int(np.round(value)) # integer

        value = np.clip(value, 1, b_max)

        self.cam.sdk.set_binning(value, self.GetVerticalBin())
        self._binning_x = self.cam.sdk.get_binning()['binning x']

    def GetHorizontalBin(self):
        return self._binning_x

    def SetVerticalBin(self, value):
        b_max, step_type = self.desc['max. binning vert'], self.desc['binning vert stepping']

        if step_type == 0:
            # binary binning, round to powers of two
            value = 1<<(value-1).bit_length()
        else:
            value = int(np.round(value)) 

        value = np.clip(value, 1, b_max)

        self.cam.sdk.set_binning(self.GetHorizontalBin(), value)
        self._binning_y = self.cam.sdk.get_binning()['binning y']

    def GetVerticalBin(self):
        return self._binning_y

    def GetSupportedBinning(self):
        import itertools

        bx_max, bx_step_type = self.desc['max. binning horizontal'], self.desc['binning horizontal stepping']
        by_max, by_step_type = self.desc['max. binning vert'], self.desc['binning vert stepping']

        if bx_step_type == 0:
            # binary step type
            x = [2**j for j in np.arange((bx_max).bit_length())]
        else:
            x = [j for j in np.arange(bx_max)]

        if by_step_type == 0:
            y = [2**j for j in np.arange((by_max).bit_length())]
        else:
            y = [j for j in np.arange(by_max)]

        return list(itertools.product(x,y))

    def SetROI(self, x0, y0, x1, y1):
        # Stepping (n pixels)
        dx = self.desc['roi hor steps']
        dy = self.desc['roi vert steps']

        # Chip size
        lx_max = self.GetCCDWidth()
        ly_max = self.GetCCDHeight()

        # Minimum ROI size
        lx_min = self.desc['min size horz']
        ly_min = self.desc['min size vert']

        # Make sure bounds are ordered
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        # Don't let the ROI go out of bounds
        # See pco.sdk manual chapter 3: IMAGE AREA SELECTION (ROI)
        x0 = np.clip(x0, 1, lx_max-dx+1)
        y0 = np.clip(y0, 1, ly_max-dy+1)
        x1 = np.clip(x1, 1+dx, lx_max)
        y1 = np.clip(y1, 1+dy, ly_max)

        # Don't let us choose too small an ROI
        if (x1-x0+1) < lx_min:
            logger.debug('Selected ROI width is too small, automatically adjusting to {}.'.format(lx_min))
            guess_pos = x0+lx_min
            # Deal with boundaries
            if guess_pos <= lx_max:
                x1 = guess_pos
            else:
                x0 = x1-lx_min-1
        if (y1-y0+1) < ly_min:
            logger.debug('Selected ROI height is too small, automatically adjusting to {}.'.format(ly_min))
            guess_pos = y0+ly_min
            if guess_pos <= ly_max:
                y1 = guess_pos
            else:
                y0 = y1-ly_min-1

        # Round to a multiple of dx, dy
        # TODO: Why do I need the 1 correction only on x0, y0???
        x0 = 1+int(np.floor((x0-1)/dx)*dx)
        y0 = 1+int(np.floor((y0-1)/dy)*dy)
        x1 = int(np.floor(x1/dx)*dx)
        y1 = int(np.floor(y1/dy)*dy)

        self.cam.sdk.set_roi(x0, y0, x1, y1)
        self._roi = [x0, y0, x1, y1]

        logger.debug('ROI set: x0 %3.1f, y0 %3.1f, w %3.1f, h %3.1f' % (x0, y0, x1-x0+1, y1-y0+1))

    def GetROI(self):
        return self._roi
    
    def GetElectrTemp(self):
        return self._electr_temp

    def GetCCDTemp(self):
        return self._ccd_temp

    def GetCCDTempSetPoint(self):
        return self._ccd_temp_set_point

    def SetCCDTemp(self, temp):
        lb = self.desc['Min Cool Set DESC']
        ub = self.desc['Max Cool Set DESC']
        temp = np.clip(temp, lb, ub)

        self.cam.sdk.set_cooling_setpoint_temperature(temp)
        self._ccd_temp_set_point = temp
        self._get_temps()

    def _get_temps(self):
        # NOTE: temperature only gets probed when acquisition starts/stops (which can be fairly
        # irregularly - to the point of not being useful).
        # TODO - find a way to safely call this while the camera is running
        d = self.cam.sdk.get_temperature()
        self._ccd_temp = d['sensor temperature']
        self._electr_temp = d['camera temperature']  # FIXME: should this be 'power temperature'?

    def GetAcquisitionMode(self):
        return self._mode

    def SetAcquisitionMode(self, mode):
        if mode in [self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT]:
            self._mode = mode
        else:
            raise RuntimeError('Mode %d not supported' % mode)

    def StartExposure(self):
        self.StopAq()
        self._get_temps()

        d = self.cam.sdk.get_delay_exposure_time()
        self._integ_time = d['exposure']*timebase[d['exposure timebase']] 
        self._cycle_time = self._integ_time \
                           + d['delay']*timebase[d['delay timebase']]
        if self._mode == self.MODE_SINGLE_SHOT:
            self.cam.record(number_of_images=1, mode='sequence')
        elif self._mode == self.MODE_CONTINUOUS:
            # Allocate buffer (2 seconds of buffer)
            self.buffer_size = int(max(int(2.0*self.GetFPS()), 1))
            self.cam.record(number_of_images=self.buffer_size, mode='ring buffer')
        self.recording = True

        self._log_exposure_start()

        return 0

    def StopAq(self):
        self.recording = False
        if self.cam.rec.recorder_handle.value is not None:
            # NOTE: probably don't need this if statement thanks to the self.recording flag,
            # but better to be safe
            self.cam.rec.stop_record()
            self.cam.rec.delete()

        self._integ_time = 0
        self._cycle_time = 0
        self.buffer_size = 0
        self.n_read = 0
        
        # read the temp in StopAq too, as this has the side effect of making sure we have
        # a reasonably correct temperature in acquisition metadata (as we
        # stop acquisition, record metadata, and then restart)
        self._get_temps()

    def GetNumImsBuffered(self):
        if self.recording:
            # FIXME: dwProcImgCount is an unsigned 32-bit integer, so this will 
            # fail after 2**32 images
            n_buf = self.cam.rec.get_status()['dwProcImgCount'] - self.n_read
        else:
            n_buf = 0
        # print(n_buf+self.n_read, self.n_read)
        return n_buf

    def GetBufferSize(self):
        # return self.cam.rec.get_settings()['maximum number of images']
        return self.buffer_size

    def GetFPS(self):
        if self.GetCycleTime() == 0:
            return 0
        return 1.0/self.GetCycleTime()

    def SetHotPixelCorrectionMode(self, mode):
        if mode in ['on', 'off']:
            self.cam.sdk.set_hot_pixel_correction_mode(mode)
        else:
            logger.debug('Hot Pixel Correction not set. Invalid mode {}. Choose "on" or "off".'.format(mode))

    def Shutdown(self):
        self.cam.close()
