'''
Alternative bindings for IDS ueye cameras using the pyueye module.

See also the uc480 module which talks directly to the ueye DLLs without an intermediate python shim. The uc480
implementation has been more widely used and tested and might be more complete. At present, however, there might be issues using
uc480 with the most recent versions of the ueye SDK, hence this module. As we don't want to maintain 2 drivers
for the same camera, this module might disappear once api compatibility issues in uc480 are resolved (i.e. use
uc480 instead of this if at all possible).

'''
from PYME.Acquire.Hardware.Camera import Camera
from pyueye import ueye
import ctypes
import threading
import logging
import queue
import numpy as np
import time
from PYME.Acquire import eventLog as event_log


logger = logging.getLogger(__name__)

def GetError(camera_handle):
    error = ctypes.c_int()
    error_message = ctypes.c_char_p()
    ueye.is_GetError(camera_handle, error, error_message)
    return error.value, error_message.value

ROI_LIMITS = {
    'UI306x' : {
        'xmin' : 96,
        'xstep' : 8,
        'ymin' : 32,
        'ystep' : 2
    },
    'UI327x' : {
        'xmin' : 256,
        'xstep' : 8,
        'ymin' : 2,
        'ystep' : 2
    },
    'UI324x' : {
        'xmin' : 16,
        'xstep' : 4,
        'ymin' : 4,
        'ystep' : 2
    }
}

# this info is partly from the IDS datasheets that one can request for each camera model
BaseProps = {
    'UI306x' : {
        # from Steve Hearn (IDS)
        #    The default gain of that camera is the absolute minimum gain the camera can deliver.
        #    All other gain factors are higher than that. This means, the system gain of 0.125 DN
        #    per electron, as specified in the camera test sheet, is the smallest possible value.
        'ElectronsPerCount'  : 7.97,
        'ReadNoise' : 6.0,
        'ADOffset' : 10
    },
    'default' : { # fairly arbitrary values
        'ElectronsPerCount'  : 10,
        'ReadNoise' : 20,
        'ADOffset' : 10
    }
}


class UEyeCamera(Camera):
    def __init__(self, device_number=0, nbits=8):
        Camera.__init__(self)
        
        self.initialized = False
        
        if nbits not in (8, 10, 12):
            raise RuntimeError('Supporting only 8, 10 or 12 bit depth, requested %d bit' % (nbits))
        self.nbits = nbits

        self.h = ueye.HIDS(device_number)

        self.check_success(ueye.is_InitCamera(self.h, None))
        self.initialized = True

        # get serial number
        cam_info = ueye.CAMINFO()
        self.check_success(ueye.is_GetCameraInfo(self.h, cam_info))
        self._serno = cam_info.SerNo.decode()
        
        # get chip size
        sensor_info = ueye.SENSORINFO()
        self.check_success(ueye.is_GetSensorInfo(self.h, sensor_info))
        
        self._chip_size = (sensor_info.nMaxWidth, sensor_info.nMaxHeight)
        self.sensor_type = sensor_info.strSensorName.decode().split('x')[0] + 'x'

        # work out the camera base parameters for this sensortype
        self.baseProps = BaseProps.get(self.sensor_type,BaseProps['default'])
                
        self.SetROI(0, 0, self._chip_size[0], self._chip_size[1])
        
        self.check_success(ueye.is_SetColorMode(self.h, getattr(ueye, 
                                                                'IS_CM_MONO%d' % self.nbits)))
        self.SetAcquisitionMode(self.MODE_CONTINUOUS)
        self._buffers = []
        self.full_buffers = queue.Queue()
        self.free_buffers = None
        
        self.n_full = 0
        
        self.n_accum = 1
        self.n_accum_current = 0
        
        self.SetIntegTime(0.1)
        self.Init()
    
    def check_success(self, function_return):
        if function_return == ueye.IS_NO_SUCCESS:
            error, message = GetError(self.h)
            raise RuntimeError('Error %d: %s' % (error, message))
        
    def Init(self):        
        self._poll = False
        self.poll_loop_active = True
        self.poll_thread = threading.Thread(target=self._poll_loop)
        self.poll_thread.start()
    
    def GetCCDWidth(self):
        return self._chip_size[0]

    def GetCCDHeight(self):
        return self._chip_size[1]
        
    def InitBuffers(self, n_buffers=50, n_accum_buffers=50):
        for ind in range(n_buffers):
            data = ueye.c_mem_p()
            buffer_id = ueye.int()

            if self.nbits == 8:
                bitsperpix = 8
                bufferdtype = np.uint8
            else: # 10 & 12 bits
                bitsperpix = 16
                bufferdtype = np.uint16

            self.check_success(ueye.is_AllocImageMem(self.h, self.GetPicWidth(), self.GetPicHeight(), bitsperpix, data, buffer_id))
            self.check_success(ueye.is_AddToSequence(self.h, data, buffer_id))
            
            self._buffers.append((buffer_id, data))

        # IDS currently only supports nMode = 0
        self.check_success(ueye.is_InitImageQueue(self.h, 0))
        
        self.transfer_buffer = np.zeros([self.GetPicHeight(), self.GetPicWidth()], bufferdtype)
        
        self.free_buffers = queue.Queue()
        # CS: we leave this as uint16 regardless of 8 or 12 bits for now as accumulation
        #     of the underlying 12 bit data should be ok (but maybe not?)
        for ind in range(n_accum_buffers):
            self.free_buffers.put(np.zeros([self.GetPicHeight(), 
                                            self.GetPicWidth()], np.uint16))
        self.accum_buffer = self.free_buffers.get()
        self.n_accum_current = 0
        self._poll = True
        
    def DestroyBuffers(self):
        self._poll = False
        self.check_success(ueye.is_ExitImageQueue(self.h))
        self.check_success(ueye.is_ClearSequence(self.h))
        
        while len(self._buffers) > 0:
            buffer_id, data = self._buffers.pop()
            self.check_success(ueye.is_FreeImageMem(self.h, data, buffer_id))
            
        self.free_buffers = None
        self.n_full = 0

    def StartExposure(self):
        logger.debug('StartAq')
        if self._poll:
            # stop, we'll allocate buffers and restart
            self.StopAq()
        # allocate at least 2 seconds of buffers
        buffer_size = int(max(2 * self.GetFPS(), 50))
        self.InitBuffers(buffer_size, buffer_size)

        event_log.logEvent('StartAq', '')
        if self._cont_mode:
            self.check_success(ueye.is_CaptureVideo(self.h, ueye.IS_DONT_WAIT))
        else:
            self.check_success(ueye.is_FreezeVideo(self.h, ueye.IS_DONT_WAIT))
        return 0
    
    def SetAcquisitionMode(self, mode):
        if mode == self.MODE_SINGLE_SHOT:
            self._cont_mode = False
        else:
            self._cont_mode = True
            
    def GetAcquisitionMode(self):
        if self._cont_mode:
            return self.MODE_CONTINUOUS
        else:
            return self.MODE_SINGLE_SHOT

    def ExpReady(self):
        return (self.full_buffers is not None) and (not self.full_buffers.empty())
        
    def ExtractColor(self, ch_slice, mode):
        buf = self.full_buffers.get()
        ch_slice[:] = buf.T
        if self.free_buffers is not None:
            # recycle buffer
            self.free_buffers.put(buf)
        self.n_full -= 1

    def StopAq(self):
        self.check_success(ueye.is_StopLiveVideo(self.h, ueye.IS_WAIT))
        self.DestroyBuffers()
    
    def _poll_buffer(self):
        data = ueye.c_mem_p()
        buffer_id = ueye.int()
        
        try:
            self.check_success(ueye.is_WaitForNextImage(self.h, 1000, data,
                                                        buffer_id))
            self.check_success(ueye.is_CopyImageMem(self.h, data, buffer_id, 
                                                    self.transfer_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))))
        except RuntimeError:
            try:
                self.check_success(ueye.is_UnlockSeqBuf(self.h, ueye.IS_IGNORE_PARAMETER, data))
            except:
                pass
            finally:
                return
        
        if self.n_accum_current == 0:
            self.accum_buffer[:] = self.transfer_buffer
        else:
            self.accum_buffer[:] = self.accum_buffer + self.transfer_buffer
        self.n_accum_current += 1
        
        self.check_success(ueye.is_UnlockSeqBuf(self.h, ueye.IS_IGNORE_PARAMETER, data))
        
        if self.n_accum_current >= self.n_accum:    
            self.full_buffers.put(self.accum_buffer)
            self.accum_buffer = self.free_buffers.get()
            self.n_accum_current = 0
            self.n_full += 1
        
    def _poll_loop(self):
        while self.poll_loop_active:
            if self._poll: # only poll if an acquisition is running
                try:
                    self._poll_buffer()
                except Exception as e:
                    logger.exception(str(e))
            else:
                time.sleep(.05)

    def CamReady(self):
        """
        Returns true if the camera is ready (initialized) not really used for
        anything, but might still be checked.

        Returns
        -------
        bool
            Is the camera ready?
        """
    
        return self.initialized
    
    def SetIntegTime(self, integ_time):
        """
        Sets the exposure time in s. Currently assumes that we will want to go as fast as possible at this exposure time
        and also sets the frame rate to match.

        Parameters
        ----------
        iTime : float
            Exposure time in s

        Returns
        -------
        None

        See Also
        --------
        GetIntegTime
        """
        new_fps = ueye.double()
        self.check_success(ueye.is_SetFrameRate(self.h, 1 / integ_time, 
                                                new_fps))
        # by default, set exposure time to max for this frame rate
        # "If 0 is passed, the exposure time is set to the maximum value of 1/frame rate."
        exposure = ueye.double(0)
        self.check_success(ueye.is_Exposure(self.h, 
                                            ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                                            exposure, ueye.sizeof(exposure)))

    def GetIntegTime(self):
        """
        Get Camera object integration time.

        Returns
        -------
        float
            The exposure time in s

        See Also
        --------
        SetIntegTime
        """
        exposure = ueye.double()
        self.check_success(ueye.is_Exposure(self.h, 
                                            ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                                            exposure, ueye.sizeof(exposure)))
        return exposure.value / 1e3


    def GetCycleTime(self):
        """
        Get camera cycle time (1/fps) in seconds (float)

        Returns
        -------
        float
            Camera cycle time (seconds)
        """
        return 1 / self.GetFPS()

    def GetPicWidth(self):
        """
        Returns the width (in pixels) of the currently selected ROI.

        Returns
        -------
        int
            Width of ROI (pixels)
        """
        x0, _, x1, _ = self.GetROI()
        return x1 - x0

    def GetPicHeight(self):
        """
        Returns the height (in pixels) of the currently selected ROI
        
        Returns
        -------
        int
            Height of ROI (pixels)
        """
        _, y0, _, y1 = self.GetROI()
        return y1 - y0

    def SetROI(self, x1, y1, x2, y2):
        """
        Set the ROI via coordinates (as opposed to via an index).

        Parameters
        ----------
        x1 : int
            Left x-coordinate, zero-indexed
        y1 : int
            Top y-coordinate, zero-indexed
        x2 : int
            Right x-coordinate, (excluded from ROI)
        y2 : int
            Bottom y-coordinate, (excluded from ROI)

        Returns
        -------
        None


        """
        logger.debug('setting ROI: %d, %d, %d, %d' % (x1, y1, x2, y2))
        limits = ROI_LIMITS[self.sensor_type]
        x1 = max(x1, limits['xmin'])
        y1 = max(y1, limits['ymin'])
        x2 = min(x2, self.GetCCDWidth())
        y2 = min(y2, self.GetCCDHeight())
        
        x_change = (x2 - x1) % limits['xstep']
        y_change = (y2 - y1) % limits['ystep']
        x2 -= x_change
        y2 -= y_change
        logger.debug('adjusted ROI: %d, %d, %d, %d' % (x1, y1, x2, y2))

        aoi = ueye.IS_RECT()
        aoi.s32X = ueye.int(x1)
        aoi.s32Y = ueye.int(y1)
        aoi.s32Width = ueye.int(x2 - x1)
        aoi.s32Height = ueye.int(y2 - y1)
        
        self.check_success(ueye.is_AOI(self.h, ueye.IS_AOI_IMAGE_SET_AOI, aoi,
                                       ueye.sizeof(aoi)))
        # have to set the integration time explicitly after changing AOI
        self.SetIntegTime(self.GetIntegTime())

        
    def GetROI(self):
        """
        
        Returns
        -------
        
            The ROI, [x1, y1, x2, y2] in the numpy convention used by SetROI

        """
        aoi = ueye.IS_RECT()
        self.check_success(ueye.is_AOI(self.h, ueye.IS_AOI_IMAGE_GET_AOI, aoi,
                                       ueye.sizeof(aoi)))
        x0, y0 = aoi.s32X.value, aoi.s32Y.value
        return x0, y0, x0 + aoi.s32Width.value, y0 + aoi.s32Height.value

    
    def GetNumImsBuffered(self):
        """
        Return the number of images in the buffer.

        Returns
        -------
        int
            Number of images in buffer
        """
        return self.n_full

    def GetBufferSize(self):
        """
        Return the total size of the buffer (in images).

        Returns
        -------
        int
            Number of images that can be stored in the buffer.
        """
        return len(self._buffers)
    
    def GetCCDTemp(self):
        di =  self._GetDeviceInfo()
        tword = di.infoDevHeartbeat.wTemperature.value
        # from IDS docs:
        #    wTemperature
        #        Camera temperature in degrees Celsius
        #        Bits 15: algebraic sign
        #        Bits 14...11: filled according to algebraic sign
        #        Bits 10...4: temperature (places before the decimal point)
        #        Bits 3...0: temperature (places after the decimal point)
        tempfloat = 1.0*(tword >> 4 & 0b1111111) + 0.1 * (tword & 0b1111)
        if (tword >> 15):
            tempfloat = -1.0 * tempfloat
        return tempfloat
    
    @property
    def noise_properties(self):
        return {'ElectronsPerCount': self.baseProps['ElectronsPerCount']/self.GetGainFactor(),
                'ReadNoise': self.baseProps['ReadNoise'],
                'ADOffset': self.baseProps['ADOffset'],
                'SaturationThreshold': 2 ** self.nbits  - 1}

    
    # @property
    # def noise_properties(self):
    #     """

    #             Returns
    #             -------

    #             a dictionary with the following entries:

    #             'ReadNoise' : camera read noise as a standard deviation in units of photoelectrons (e-)
    #             'ElectronsPerCount' : AD conversion factor - how many electrons per ADU
    #             'NoiseFactor' : excess (multiplicative) noise factor 1.44 for EMCCD, 1 for standard CCD/sCMOS. See
    #                 doi: 10.1109/TED.2003.813462

    #             and optionally
    #             'ADOffset' : the dark level (in ADU)
    #             'DefaultEMGain' : a sensible EM gain setting to use for localization recording
    #             'SaturationThreshold' : the full well capacity (in ADU)

    #             """
        
    #     return {
    #         'ReadNoise': 1,
    #         'ElectronsPerCount': 1,
    #         'NoiseFactor': 1,
    #         'ADOffset': 0,
    #         'SaturationThreshold': 2 ** self.nbits  - 1
    #     }
    
    def GetFPS(self):
        """
        Get the camera frame rate in frames per second (float).

        Returns
        -------
        float
            Camera frame rate (frames per second)
        """
        fps = ueye.double()
        self.check_success(ueye.is_GetFramesPerSecond(self.h, fps))
        return fps.value

    def GetSerialNumber(self):
        return self._serno

    def GetName(self):
        return 'ueye-camera'

    def GetHeadModel(self):
        return self.sensor_type

    def SetGain(self, gain=100):
        self.check_success(ueye.is_SetHardwareGain(self.h, gain, ueye.IS_IGNORE_PARAMETER,
                                                   ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER))
        
    def GetGain(self):
        ret = ueye.is_SetHardwareGain(self.h, ueye.IS_GET_MASTER_GAIN,
                                      ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)
        return ret
    
    def GetGainFactor(self):
        gain = self.GetGain()
        ret = ueye.is_SetHWGainFactor(self.h, ueye.IS_INQUIRE_MASTER_GAIN_FACTOR, gain)
        return 0.01*ret

    #### Some extra functions for this camera

    def _GetDeviceInfo(self):
        dev_info = ueye.IS_DEVICE_INFO()
        self.check_success(ueye.is_DeviceInfo(self.h,ueye.IS_DEVICE_INFO_CMD_GET_DEVICE_INFO,
                                              dev_info,ueye.sizeof(dev_info)))
        return dev_info
