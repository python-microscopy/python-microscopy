from . import Camera
import six
import atexit
import ctypes
import sys

if sys.platform == 'win32':
    memcpy = ctypes.cdll.msvcrt.memcpy
elif sys.platform == 'darwin':
    memcpy = ctypes.CDLL('libSystem.dylib').memcpy
else: #linux
    memcpy = ctypes.CDLL('libc.so.6').memcpy

from thorlabs_tsi_sdk import tl_camera
sdk = tl_camera.TLCameraSDK()
atexit.register(sdk.dispose)

available_cameras = sdk.discover_available_cameras()

class ThorlabsCamera(Camera.Camera):
    def __init__(self, which=0, buffer_size=100):
        """
        Parameters
        ----------
        
        which : str, int, or none
            Which camera to open - either a serial number, an index into available_cameras
        """
        Camera.Camera.__init__(self)
        
        if isinstance(which, six.string_types):
            if not which in available_cameras:
                raise RuntimeError('Camera with serial number %s is not available' % which)
            self._serial_number = which
        else:
            self._serial_number = available_cameras[which]
            
        self._cam = sdk.open_camera(self._serial_number)
        self._cam.image_poll_timeout_ms = 50
            
        self.SetAcquisitionMode(self.MODE_CONTINUOUS)
        
        self._buffer_size=100 # TODO - should this change with ROI size?
        
        
        
    def ExpReady(self):
        f = self._cam.get_pending_frame_or_null()
        if f is None:
            return False
        else:
            self._f = f
            return True
    
    def ExtractColor(self, chSlice, mode):
        memcpy(chSlice.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
               self._f.image_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)
        
    def GetSerialNumber(self):
        return self._serial_number
    
    def GetIntegTime(self):
        return self._cam.exposure_time_us/1.0e6
    
    def SetIntegTime(self, iTime):
        self._cam.exposure_time_us = int(iTime*1.0e6)
        
    def GetCycleTime(self):
        return 1.0/self._cam.get_measured_frame_rate_fps()
    
    def GetFPS(self):
        return self._cam.get_measured_frame_rate_fps()
        
    def GetCCDWidth(self):
        return self._cam.sensor_width_pixels
    
    def GetCCDHeight(self):
        return self._cam.sensor_height_pixels
    
    def GetPicWidth(self):
        return self._cam.image_width_pixels
    
    def GetPicHeight(self):
        return self._cam.image_height_pixels
    
    def GetHorizontalBin(self):
        return self._cam.binx
    
    def GetVerticalBin(self):
        return self._cam.biny
    
    def SetROI(self, x1, y1, x2, y2):
        self._cam.roi = tl_camera.ROI(x1, y1, x2-1, y2-1)
        
    def GetROI(self):
        x1, y1, x2, y2 = self._cam.roi
        return (x1, y1, x2+1, y2+1)
    
    def GetAcquisitionMode(self):
        return self._aq_mode
    
    def SetAcquisitionMode(self, mode):
        if (mode == self.MODE_CONTINUOUS):
            self._aq_mode = mode
            self._cam.frames_per_trigger_zero_for_unlimited = 0
            self._cam.operation_mode = tl_camera.OPERATION_MODE.SOFTWARE_TRIGGERED
        else:
            self._aq_mode = mode
            self._cam.frames_per_trigger_zero_for_unlimited = 1
            if (mode==self.MODE_HARDWARE_TRIGGER):
                self._cam.operation_mode = tl_camera.OPERATION_MODE.HARDWARE_TRIGGERED
            else:
                self._cam.operation_mode = tl_camera.OPERATION_MODE.SOFTWARE_TRIGGERED
                
    def StartExposure(self):
        self._cam.arm(self._buffer_size)
        
        if self._aq_mode  in (self.MODE_SINGLE_SHOT, self.MODE_CONTINUOUS):
            self._cam.issue_software_trigger()
            
    def StopAq(self):
        self._cam.disarm()
        
    def Shutdown(self):
        self._cam.dispose()
        
    def GetBufferSize(self):
        return self._buffer_size
    
    def GetNumImsBuffered(self):
        return 1 # FIXME! - this is a fudge which should get things working for now
    
    def GetCCDTemp(self):
        return 27.0 #FIXME - Fake value as a fudge
        
            