
from PYME.Acquire.Hardware.Camera import Camera
from ids_peak import ids_peak as peak
import logging
import numpy as np
import threading
import time
import ctypes
import queue
from PYME.Acquire import eventLog as event_log
from threading import Lock

# The IDS Peak API supports uEye+ cameras (U3/GV models) as well as "almost all" uEye (UI models) 
# to use this implementation, first install the IDS Peak SDK (our interface developed on version 2.6.2.0)
# You'll have to download the IDS Peak SDK from the IDS website and install it.
# IDS Peak 2.10 and on: install the ids_peak python package from pypi
# https://pypi.org/project/ids-peak/
# IDS Peak <2.10: install the ids_peak python package from their wheel in your IDS peak installation folder, e.g.:
# C:\Program Files\IDS\ids_peak\sdk\api\binding\python\wheel\x86_[32|64]
# pip install ids_peak-<version>-cp<version>-cp<version>[m]-[win32|win_amd64].whl
# You can ignore the IPL / AFL wheels - these are for 'basic image processing' and 'auto features' libraries
# which we don't use. We use the generic SDK (rather than the 'comfort' SDK).
# For my installation, 64 bit windows, python 3.8, I installed ids_peak with:
# python -m pip install "C:\Program Files\IDS\ids_peak\generic_sdk\api\binding\python\wheel\x86_64\ids_peak-1.6.2.0-cp38-cp38-win_amd64.whl"

logger = logging.getLogger(__name__)

def find_ids_peak_cameras():
    peak.Library.Initialize()
    device_manager = peak.DeviceManager.Instance()
    device_manager.Update()
    return device_manager.Devices()

# at the moment, only support 'unpacked' types, i.e. 12 bit in 2 bytes, not 1.5 bytes
PIXEL_FORMATS = {  
    8: 'Mono8',
    10: 'Mono10',
    12: 'Mono12',
    # 16: 'Mono16',  # Only supported by uEye cameras (UI models, not e.g. U3)
}

class IDS_Camera(Camera):
    """

    Notes
    -----
    Camera noise_properties dictionary will be keyed off the gain_mode property,
    which corresponds to the bit-depth for this class. See uEye example in
    PYME.Acquire.Hardware.camera_noise.
    """
    
    supports_software_trigger = True

    def __init__(self, device_number=0, nbits=8):
        import sys
        self.initialized = False
        super().__init__()
        self.device_number = device_number
        self.nbits = nbits
        self.n_full = 0
        self._buffer_lock = Lock()  # use to avoid race conditions while polling buffers during their destruction

        devices = find_ids_peak_cameras()
        if len(devices) == 0:
            raise RuntimeError('No IDS peak cameras found')
        
        # open the device
        self.device = devices[device_number].OpenDevice(peak.DeviceAccessType_Control)
        self.serial_number = self.device.SerialNumber()
        self.model_name = self.device.ModelName()
        logger.info(f'IDS peak camera {self.model_name} opened, serial number {self.serial_number}')

        # get the remote device 'node map'
        self._node_map = self.device.RemoteDevice().NodeMaps()[0]

        # open a datastream
        # self._data_stream = self.device.DataStreams()[0]
        self._data_stream = self.device.DataStreams()[0].OpenDataStream()
        self._data_stream_node_map = self._data_stream.NodeMaps()[0]
        # set to FIFO
        self._data_stream_node_map.FindNode("StreamBufferHandlingMode").SetCurrentEntry("OldestFirst")

        # set ROI size to full usable sensor size ---------------------
        # get min offsets. Note that some IDS cameras do not use the full chip.
        self._offset_x_min = self._node_map.FindNode("OffsetX").Minimum()
        self._offset_y_min = self._node_map.FindNode("OffsetY").Minimum()
        self._width_min = self._node_map.FindNode("Width").Minimum()
        self._height_min = self._node_map.FindNode("Height").Minimum()
        
        # set the minimum, to remove size restrictions due to current settings
        self._node_map.FindNode("OffsetX").SetValue(self._offset_x_min)
        self._node_map.FindNode("OffsetY").SetValue(self._offset_y_min)
        self._node_map.FindNode("Width").SetValue(self._width_min)
        self._node_map.FindNode("Height").SetValue(self._height_min)

        # now query max values
        self._offset_x_max = self._node_map.FindNode("OffsetX").Maximum()
        self._offset_y_max = self._node_map.FindNode("OffsetY").Maximum()
        self._width_max = self._node_map.FindNode("Width").Maximum()
        self._height_max = self._node_map.FindNode("Height").Maximum()
        # get increments
        self._offset_x_incr = self._node_map.FindNode("OffsetX").Increment()
        self._offset_y_incr = self._node_map.FindNode("OffsetY").Increment()
        self._width_increment = self._node_map.FindNode("Width").Increment()
        self._height_increment = self._node_map.FindNode("Height").Increment()

        # set ROI to full sensor size:
        self._node_map.FindNode("OffsetX").SetValue(self._offset_x_min)
        self._node_map.FindNode("OffsetY").SetValue(self._offset_y_min)
        self._node_map.FindNode("Width").SetValue(self._width_max)
        self._node_map.FindNode("Height").SetValue(self._height_max)
        # ----------------------------------------------------------------

        # find out if this model supports board/sensor temperature:
        try:
            self._node_map.FindNode("DeviceTemperature")
            self._has_temperature = True
        except:
            self._has_temperature = False

        self.SetAcquisitionMode(self.MODE_CONTINUOUS)
        # allocate buffers
        # self.allocate_buffers()
        self.full_buffers = None

        self._buffer_poll_wait_time_ms = 5000  # [ms]

        # set pixel format
        self._node_map.FindNode("PixelFormat").SetCurrentEntry(PIXEL_FORMATS[nbits])
        # IDS sends all multi-byte little-endian, and we're going to copy in native computer order,
        # throw an error unless we're matched.
        assert sys.byteorder == 'little'

        self.Init()
        self.initialized = True
        
    def Init(self):        
        self._poll = False
        self.poll_loop_active = True
        self.poll_thread = threading.Thread(target=self._poll_loop)
        self.poll_thread.start()
    
    def _poll_loop(self):
        while self.poll_loop_active:
            if self._poll: # only poll if an acquisition is running
                try:
                    with self._buffer_lock:
                        self._poll_buffer()
                except Exception as e:
                    logger.exception(str(e))
            else:
                time.sleep(.05)
    
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
    
    def _poll_buffer(self):
        try:
            buffer = self._data_stream.WaitForFinishedBuffer(self._buffer_poll_wait_time_ms)
            # copy over
            ctypes.memmove(self.transfer_buffer, int(buffer.BasePtr()), int(buffer.Size()))
            # ctypes uses native byteorder, so we are OK if this is little-endian system 
            arr = np.frombuffer(self.transfer_buffer, dtype=self.transfer_buffer_dtype)
            # return camera buffer to queue
            self._data_stream.QueueBuffer(buffer)
            arr = arr.reshape((self.curr_height, self.curr_width))
            self.full_buffers.put(arr)
            self.n_full += 1
        except Exception as e:
            logger.error(f'Error polling buffer: {e}')

    
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
    
    def Close(self):
        self.StopAq()
        # self.DestroyBuffers()  # already destroyed in StopAq
        peak.Library.Close()
    
    def DestroyBuffers(self):
        with self._buffer_lock:
            self.n_full = 0

            # remove camera-side buffers
            for b in self._data_stream.AnnouncedBuffers():
                try:
                    self._data_stream.RevokeBuffer(b)
                except Exception as e:
                    logger.error(f'Error revoking buffer: {e}')
                
            # computer RAM: destroy free and full buffer queues
            while not self.full_buffers.empty():
                try:
                    self.full_buffers.get_nowait()
                except queue.Empty:
                    pass
            
            while not self.free_buffers.empty():
                try:
                    self.free_buffers.get_nowait()
                except queue.Empty:
                    pass
    
    def allocate_buffers(self, n_buffers=50):
        self._n_cam_buffers = n_buffers
        # camera side
        try:
            self._data_stream.Flush(peak.DataStreamFlushMode_DiscardAll)
            for b in self._data_stream.AnnouncedBuffers():
                self._data_stream.RevokeBuffer(b)
            # get current payload size
            self._payload_size = self._node_map.FindNode('PayloadSize').Value()
            # allocate buffers
            for ind in range(n_buffers):
                b = self._data_stream.AllocAndAnnounceBuffer(self._payload_size)
                self._data_stream.QueueBuffer(b)
        except Exception as e:
            logger.error(f'Error allocating buffers: {e}')
            raise e
        
        # computer RAM

        # transfer buffer
        if self.nbits == 8:
            bufferdtype = np.uint8
        else: # 10 & 12 bits
            bufferdtype = np.uint16
        
        self.curr_height, self.curr_width = self.GetPicHeight(), self.GetPicWidth()
        self.transfer_buffer_size = self.curr_height * self.curr_width * bufferdtype().itemsize
        self.transfer_buffer_dtype = bufferdtype
        self.transfer_buffer = ctypes.create_string_buffer(self.transfer_buffer_size)
        self.transfer_buffer_memory_v = ctypes.c_char()
        self.transfer_buffer_memory = ctypes.pointer(self.transfer_buffer_memory_v)
        self.transfer_buffer_id = ctypes.c_int()
        # others
        self.free_buffers = queue.Queue()
        self.full_buffers = queue.Queue()
        for ind in range(n_buffers):
            # Note that it is essential to use zeros here to ensure
            # compatibility with Mono8, Mono10, Mono12 written into uint16
            self.free_buffers.put(np.zeros((self.GetPicHeight(), self.GetPicWidth()), 
                                           dtype=np.uint16))
        self._poll = False
    
    def StopAq(self):
        self._poll = False
        try:
            # tell the camera to stop
            self._node_map.FindNode('AcquisitionStop').Execute()

            # stop the datastream
            self._data_stream.KillWait()  # interrupts 1 WaitForFinishedBuffer call
            self._data_stream.StopAcquisition(peak.AcquisitionStopMode_Default)
            # flush TODO - do we really want to flush immediately?
            self._data_stream.Flush(peak.DataStreamFlushMode_DiscardAll)

            # unlock parameters
            self._node_map.FindNode('TLParamsLocked').SetValue(0)
        except Exception as e:
            logger.error(f'Error stopping acquisition: {e}')
            raise e
        self.DestroyBuffers()
    
    def StartExposure(self):
        logger.debug('StartAq')
        if self._poll:
            # stop, we'll allocate buffers and restart
            self.StopAq()
        # allocate at least 2 seconds of buffers
        buffer_size = int(max(2 * self.GetFPS(), 50))
        logger.info('Allocating {} buffers'.format(buffer_size))
        self.allocate_buffers(buffer_size)

        self._log_exposure_start()
        try:
            
            if self._acq_mode == self.MODE_CONTINUOUS:
                self._node_map.FindNode('TriggerMode').SetCurrentEntry('Off')
                
            elif self._acq_mode == self.MODE_SOFTWARE_TRIGGER:
                self._node_map.FindNode('TriggerSelector').SetCurrentEntry('ExposureStart')
                self._node_map.FindNode('TriggerSource').SetCurrentEntry('Software')
                self._node_map.FindNode('TriggerMode').SetCurrentEntry('On')
                

            elif self._acq_mode == self.MODE_HARDWARE_TRIGGER:
                self._node_map.FindNode('TriggerSelector').SetCurrentEntry('ExposureStart')
                # line0 should be (trigger) input with optocoupler for all IDS
                # cameras covered in IDS peak 2.10.0 with hardware triggering
                self._node_map.FindNode('TriggerSource').SetCurrentEntry('Line0')
                self._node_map.FindNode('TriggerMode').SetCurrentEntry('On')
            else:
                raise NotImplementedError('Single shot mode not implemented')
            
            self._node_map.FindNode('TLParamsLocked').SetValue(1)
            self._data_stream.StartAcquisition(peak.AcquisitionStartMode_Default,
                                                peak.DataStream.INFINITE_NUMBER)
            self._node_map.FindNode('AcquisitionStart').Execute()

        except Exception as e:
            logger.error(f'Error starting acquisition: {e}')
            raise e
        self._poll = True
        return 0
    
    def FireSoftwareTrigger(self):
        self._node_map.FindNode('TriggerSoftware').Execute()
        self._log_exposure_start()
        # self._node_map.FineNode('TriggerSoftware').WaitUntilDone(1000)
    
    def GetIntegTime(self):
        """
        Get the current exposure time.

        Returns
        -------
        float
            The exposure time in s

        See Also
        --------
        SetIntegTime
        """
        exposure_time = self._node_map.FindNode("ExposureTime").Value()  # [us]
        return exposure_time / 1e6  # [s]
    
    def SetIntegTime(self, exposure_time):
        """
        Set the exposure time. Automatically adjusts frame rate

        Parameters
        ----------
        exposure_time : float
            The exposure time in s

        See Also
        --------
        GetIntegTime
        """
        # note that setting exposure time will automatically increase decrease
        # FPS if necessary, but setting exposure time lower will not 
        # automatically increase FPS
        
        # Exposure time min/max will auto adjust given the current frame rate,
        # so best to ignore it, and let frame rate adjust to what it can.
        # lower = self._node_map.FindNode("ExposureTime").Minimum()  # [us]
        # upper = self._node_map.FindNode("ExposureTime").Maximum()  # [us]
        # exp_time = np.clip(exposure_time * 1e6, lower, upper)  # [us]
        # logger.info(f'Setting exposure time to {exp_time} us')
        # self._node_map.FindNode("ExposureTime").SetValue(exp_time)
        self._node_map.FindNode("ExposureTime").SetValue(exposure_time * 1e6)
        fps_target = 1 / exposure_time  # [FPS]
        lower = self._node_map.FindNode("AcquisitionFrameRate").Minimum()  # [FPS]
        upper = self._node_map.FindNode("AcquisitionFrameRate").Maximum()  # [FPS]
        fps = np.clip(fps_target, lower, upper)
        self._node_map.FindNode("AcquisitionFrameRate").SetValue(fps)
    
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

    def GetNumImsBuffered(self):
        """
        Return the number of images in the buffer.

        Returns
        -------
        int
            Number of images in buffer
        """
        # can enable camera-side buffer queue monitoring, but seems to return total number
        # of frames acquired, not number of full frames waiting in the queue.
        # self._data_stream_node_map.FindNode("BufferStatusMonitoringEnabled").SetValue(True)
        # return self._data_stream_node_map.FindNode("BufferStatusOutputQueueCount").Value()
        
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
            return self._n_cam_buffers
        else:
            # if we aren't polling, spoof infinitely large buffer so we don't
            # flag a buffer overflow while we e.g. rebuild the buffers. This
            # makes no functional difference for us, but avoids a spurious
            # warning from frameWrangler about the buffer overflowing.
            return np.iinfo(np.int32).max
    
    def GetROI(self):
        """
        Returns the current ROI as a tuple of (x0, y0, x1, y1).

        Returns
        -------
        tuple
            (x0, y0, x1, y1)
        """
        x0 = self._node_map.FindNode("OffsetX").Value()
        y0 = self._node_map.FindNode("OffsetY").Value()
        x1 = x0 + self._node_map.FindNode("Width").Value()
        y1 = y0 + self._node_map.FindNode("Height").Value()
        return (x0, y0, x1, y1)
    
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

        # set the minimum, to remove size restrictions due to current settings
        self._node_map.FindNode("OffsetX").SetValue(self._offset_x_min)
        self._node_map.FindNode("OffsetY").SetValue(self._offset_y_min)
        self._node_map.FindNode("Width").SetValue(self._width_min)
        self._node_map.FindNode("Height").SetValue(self._height_min)

        # check offset increments
        x1 -= x1 % self._offset_x_incr
        y1 -= y1 % self._offset_y_incr
        # clip to bounds
        x1 = int(np.clip(x1, self._offset_x_min, self._offset_x_max))
        y1 = int(np.clip(y1, self._offset_y_min, self._offset_y_max))
        x2 = int(np.clip(x2, x1 + self._width_min, self._width_max))
        y2 = int(np.clip(y2, y1 + self._height_min, self._height_max))
        # double check width/height increments
        x2 -= (x2 - x1) % self._width_increment  # ROI must be a multiple of increment
        y2 -= (y2 - y1) % self._height_increment
        logger.debug('adjusted ROI: %d, %d, %d, %d' % (x1, y1, x2, y2))

        self._node_map.FindNode("OffsetX").SetValue(x1)
        self._node_map.FindNode("OffsetY").SetValue(y1)
        self._node_map.FindNode("Width").SetValue(x2 - x1)
        self._node_map.FindNode("Height").SetValue(y2 - y1)
        # using ueye api we used to have to set integration time after adjusting
        # ROI. Not sure if we need that here or not. Leave it for not just in case.
        self.SetIntegTime(self.GetIntegTime())

    def SetAcquisitionMode(self, mode):
        """Set a flag so next StartExposure can toggle between continuous and single shot mode.

        Parameters
        ----------
        mode : int
            toggles between continuous and single shot mode
        """
        self._acq_mode = mode
            
    def GetAcquisitionMode(self):
        return self._acq_mode
    
    def GetFPS(self):
        """
        Returns the current frame rate (in frames per second).

        Returns
        -------
        float
            Frame rate (fps)
        """
        return self._node_map.FindNode("AcquisitionFrameRate").Value()

    def GetSerialNumber(self):
        return self.serial_number
    
    def GetCCDTemp(self):
        if self._has_temperature:
            return self._node_map.FindNode("DeviceTemperature").Value()  # [degrees C]
        else:
            return 0

    def GetCycleTime(self):
        return 1 / self.GetFPS()

    def GetCCDHeight(self):
        """
        Returns
        -------
        int
            The sensor height in pixels

        """
        return self._height_max
    
    def GetCCDWidth(self):
        """
        Returns
        -------
        int
            The sensor width in pixels

        """
        return self._width_max
    
    @property
    def _gain_mode(self):
        return '%d-bit' % self.nbits
