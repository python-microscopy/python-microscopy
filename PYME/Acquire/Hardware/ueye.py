'''
Alternative bindings for IDS ueye cameras using the pyueye module.

See also the uc480 module which talks directly to the ueye DLLs without an
intermediate python shim. The uc480 implementation has been widely used with
previous ueye driver versions (e.g. <4.93). At present, there might be issues 
using uc480 with the most recent versions of the ueye SDK, hence this module.
This module works with ueye driver 4.96.1, and pyueye 4.96.952. Pyueye handles
some boiler plate ctypes code for us, but in the future we may call the dll 
directly as in uc480.

# NOTE to developers: pyueye typing is somewhat obnoxious. Some oddities with
# ctypes pointers (testing for equality of a factory-produced class) makes it
# safer to use their pointer cast method, and their 'extra functionality' ctypes

Application notes for specific cameras from the SDK:
327x: 
    Triggering: The internal sensor delay is about 2-3 lines when triggering. 
        The line period depends on the selected pixel clock. The higher the 
        pixel clock, the smaller the line period is. In overlapping trigger 
        mode, the camera timestamp may be overwritten if the maximum exposure
        time is used. In this case, reduce the exposure time by approx. 1%

'''
from PYME.Acquire.Hardware.Camera import Camera, MultiviewCameraMixin
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
    },
    'UI124x' : {
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
    'UI327x' : {  # calibrated by AESB 2022/04 on S/N 4103211322 running in 12 bit mode, 100 ms integration time.
        'ElectronsPerCount': 2.706,  # fitted from Var [ADU^2] vs Mean [ADU] plot (1/slope)
        'ReadNoise' : 2.425, # median of 100 ms varmap from gen_sCMOS_maps.py is 5.883 e-^2. ReadNoise is sigma, i.e. sqrt of that
        'ADOffset' : 7.67,  # median of 100 ms dark map from gen_sCMOS_maps.py
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
        
        self._chip_size = (int(sensor_info.nMaxWidth), int(sensor_info.nMaxHeight)) # convert from c_uint, otherwise trips up JSON dumps
        self.sensor_type = sensor_info.strSensorName.decode().split('x')[0] + 'x'

        # work out the camera base parameters for this sensortype
        self.baseProps = BaseProps.get(self.sensor_type,BaseProps['default'])
        
        # note that some uEye cameras have a sensor size which exceeds the 'usable' ROI
        self.SetROI(0, 0, self._chip_size[0], self._chip_size[1])
        
        self.check_success(ueye.is_SetColorMode(self.h, getattr(ueye, 
                                                                'IS_CM_MONO%d' % self.nbits)))
        
        # turn off hardware gamma if supported.
        hw_gamma = ueye.is_SetHardwareGamma(self.h, ueye.int(ueye.IS_GET_HW_SUPPORTED_GAMMA))
        if hw_gamma == ueye.IS_SET_HW_GAMMA_ON:  # SetHardwareGamma returns IS_SET_HW_GAMMA_ON (1) if supported
            logger.debug('model supports hardware gamma correction, turning it off')
            self.check_success(ueye.is_SetHardwareGamma(self.h, ueye.IS_SET_HW_GAMMA_OFF))

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
        if function_return != ueye.IS_SUCCESS:
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

        self.check_success(ueye.is_ImageQueue(self.h, ueye.IS_IMAGE_QUEUE_CMD_INIT, None, ctypes.c_int(0)))
        
        self.curr_height, self.curr_width = self.GetPicHeight(), self.GetPicWidth()
        self.transfer_buffer_size = self.curr_height * self.curr_width * bufferdtype().itemsize
        self.transfer_buffer_dtype = bufferdtype
        self.transfer_buffer = ctypes.create_string_buffer(self.transfer_buffer_size)
        self.transfer_buffer_memory_v = ueye.char()
        self.transfer_buffer_memory = ueye._pointer_cast(self.transfer_buffer_memory_v, ueye.char_p)
        self.transfer_buffer_id = ueye.int()
        self.wait_buffer = ueye.IMAGEQUEUEWAITBUFFER()
        self.wait_buffer.timeout = ueye.uint(1000)
        
        self.wait_buffer.pnMemId = ueye._pointer_cast(self.transfer_buffer_id, ctypes.POINTER(ueye.int))
        self.wait_buffer.ppcMem = ueye._pointer_cast(self.transfer_buffer_memory, ctypes.POINTER(ueye.char_p))
        
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
        self._poll = False  # already in StopAq, can probably remove
        self.n_full = 0
        # exit the image queue    
        self.check_success(ueye.is_ImageQueue(self.h, ueye.IS_IMAGE_QUEUE_CMD_EXIT, None, ueye.int(0)))
        
        # remove all image memories from the sequence list we created
        self.check_success(ueye.is_ClearSequence(self.h))
        
        # free up each image memory we allocated on the device
        while len(self._buffers) > 0:
            buffer_id, data = self._buffers.pop()
            self.check_success(ueye.is_FreeImageMem(self.h, data, buffer_id))
            
        # destroy free buffers and remove queue of full ones
        self.free_buffers = None
        while not self.full_buffers.empty():
            try:
                self.full_buffers.get_nowait()
            except queue.Empty:
                pass

    def StartExposure(self):
        logger.debug('StartAq')
        if self._poll:
            # stop, we'll allocate buffers and restart
            self.StopAq()
        # allocate at least 2 seconds of buffers
        buffer_size = int(max(2 * self.GetFPS(), 50))
        self.InitBuffers(buffer_size, buffer_size)

        self._log_exposure_start()
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
        return (self.full_buffers is not None) and (self.n_full > 0)
        
    def ExtractColor(self, ch_slice, mode):
        # get nowait to hard-throw an Empty error if we've entered this method
        # and we shouldn't have
        buf = self.full_buffers.get_nowait()
        ch_slice[:] = buf.T
        if self.free_buffers is not None:
            # recycle buffer
            self.free_buffers.put(buf)
        self.n_full -= 1

    def StopAq(self):
        self._poll = False
        # cancel any ongoing waits (e.g. shutdown)
        self.check_success(ueye.is_ImageQueue(self.h, ueye.IS_IMAGE_QUEUE_CMD_CANCEL_WAIT, None, ueye.int(0)))
        self.check_success(ueye.is_StopLiveVideo(self.h, ueye.IS_WAIT))# ueye.IS_FORCE_VIDEO_STOP))
        self.DestroyBuffers()
    
    def _poll_buffer(self):
        try:
            # Query the ID/location of earlier frame in the ring buffer, or wait
            # for the next one. Will fill wait_buffer.nMemId and pcMem
            self.check_success(ueye.is_ImageQueue(self.h, ueye.IS_IMAGE_QUEUE_CMD_WAIT, 
                                                  self.wait_buffer, ueye.sizeof(self.wait_buffer)))
            # self.check_success(ueye.is_CopyImageMem(self.h, self.wait_buffer.ppcMem, self.transfer_buffer_id, 
            #                                         self.transfer_buffer))#.ctypes.data_as(ctypes.POINTER(ueye.char_p))))
                                                    # self.transfer_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
            ctypes.memmove(self.transfer_buffer, self.wait_buffer.ppcMem.contents, self.transfer_buffer_size)
            arr = np.frombuffer(self.transfer_buffer, dtype=self.transfer_buffer_dtype)
            arr = arr.reshape((self.curr_height, self.curr_width))
        except RuntimeError as e:
            if 'Error %d' % ueye.IS_OPERATION_ABORTED in str(e):
                # we are shutting down the camera, let the other thread handle
                # clean-up
                logger.debug('ImageQueue wait canceled, returning')
                return
            logger.error(e)
            try:
                self.check_success(ueye.is_UnlockSeqBuf(self.h, self.transfer_buffer_id, None))
            except Exception as e:
                logger.error(e)
            finally:
                return
        
        if self.n_accum_current == 0:
            self.accum_buffer[:] = arr
        else:
            self.accum_buffer[:] = self.accum_buffer + arr
        self.n_accum_current += 1
        
        self.check_success(ueye.is_UnlockSeqBuf(self.h, self.transfer_buffer_id, None))
        
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
        if self._poll:
            return len(self._buffers)
        else:
            # if we aren't polling, spoof infinitely large buffer so we don't
            # flag a buffer overflow while we e.g. rebuild the buffers. This
            # makes no functional difference for us, but avoids a spurious
            # warning from frameWrangler about the buffer overflowing.
            return np.iinfo(np.int32).max
    
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
        try:  # try and get noise properties following the current convention
            return super().noise_properties
        except RuntimeError:  # fall back loudly on "base properties"
            logger.exception('Noise properties not set up for this camera, falling back on values which are likely wrong')
            return {'ElectronsPerCount': self.baseProps['ElectronsPerCount']/self.GetGainFactor(),
                    'ReadNoise': self.baseProps['ReadNoise'],
                    'ADOffset': self.baseProps['ADOffset'],
                    'SaturationThreshold': 2 ** self.nbits  - 1}
    
    @property
    def _gain_mode(self):
        return '%d-bit' % self.nbits

    
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
    
    def SetOutputTrigger(self, mode, delay=0, width=0.0001, positive=True):
        """
        Set output trigger of the camera. Currently hard-coded to use GPIO1

        Parameters
        ----------
        mode : str
            Currently supported modes include:
                low: constant TTL low
                high: constant TTL high
                flash: goes TTL high during exposure. If 0 is passed, the flash
                    output will be active until the end of the exposure time.
                    With global start shutter this is the time until the end of
                    exposure for the first row.
        delay : float, optional
            delay after trigger event, in seconds, to emit TTL high (assuming 
            posiive polarity), by default 0 s.
        width : float, optional
            TTL high pulse width, in seconds, by default 0.0001 s, or 0.1 ms
        positive : bool, optional
            CURRENTLY IGNORED Sets polarity of the output trigger to positive 
            (True) or negative (False). True, by default.
        
        """
        # set GPIO1 to be output flash
        m = ueye.int(ueye.IO_FLASH_MODE_GPIO_1)
        self.check_success(ueye.is_IO(self.h,
                                          ueye.IS_IO_CMD_FLASH_SET_MODE,
                                          m, ueye.sizeof(m)))
        
        if mode == 'high':
            mode = ueye.int(ueye.IO_FLASH_MODE_CONSTANT_HIGH)
            self.check_success(ueye.is_IO(self.h,
                                          ueye.IS_IO_CMD_FLASH_SET_MODE,
                                          mode, ueye.sizeof(mode)))
            return
        elif mode == 'low':
            mode = ueye.int(ueye.IO_FLASH_MODE_CONSTANT_LOW)
            self.check_success(ueye.is_IO(self.h,
                                          ueye.IS_IO_CMD_FLASH_SET_MODE,
                                          mode, ueye.sizeof(mode)))
            return

        if mode =='flash':
            mode = ueye.int(ueye.IO_FLASH_MODE_FREERUN_HI_ACTIVE)
            self.check_success(ueye.is_IO(self.h,
                                          ueye.IS_IO_CMD_FLASH_SET_MODE,
                                          mode, ueye.sizeof(mode)))
        else:
            raise RuntimeError('Unsupported output trigger mode: %s' % mode)
        
        fp = ueye.IO_FLASH_PARAMS()
        fp.s32Delay = ueye.c_int(int(delay * 1e6))  # [s] -> [us]
        fp.u32Duration = ueye.c_uint(int(width * 1e6))  # [s] -> [us]
        self.check_success(ueye.is_IO(self.h,
                                          ueye.IS_IO_CMD_FLASH_SET_PARAMS,
                                          fp, ueye.sizeof(fp)))

    #### Some extra functions for this camera

    def _GetDeviceInfo(self):
        dev_info = ueye.IS_DEVICE_INFO()
        self.check_success(ueye.is_DeviceInfo(self.h,ueye.IS_DEVICE_INFO_CMD_GET_DEVICE_INFO,
                                              dev_info,ueye.sizeof(dev_info)))
        return dev_info
    
    def GetGlobalFlashSettings(self):
        """Query current global 'flash', i.e. output trigger settings

        Returns
        -------
        delay: float
            [s]
        width: float
            duration of TTL in [s]
        
        """
        fp = ueye.IO_FLASH_PARAMS()
        self.check_success(ueye.is_IO(self.h, 
                                      ueye.IS_IO_CMD_FLASH_GET_GLOBAL_PARAMS,
                                      fp, ueye.sizeof(fp)))
        return fp.s32Delay.value / 1e6, fp.u32Duration.value / 1e6


#TODO - replace MultiviewCameraMixin with a Multiview wrapper so that we don't need to have explicit multiview versions of all cameras.
class MultiviewUEye(MultiviewCameraMixin, UEyeCamera):
    def __init__(self, camNum, multiview_info, nbits=8):
        UEyeCamera.__init__(self, camNum, nbits)
        # default to the whole chip        
        default_roi = dict(xi=0, xf=int(self._chip_size[0]), yi=0, yf=int(self._chip_size[1]))
        MultiviewCameraMixin.__init__(self, multiview_info, default_roi, UEyeCamera)
