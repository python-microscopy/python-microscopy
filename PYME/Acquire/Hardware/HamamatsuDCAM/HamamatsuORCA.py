#!/usr/bin/python

###############
# HamamatsuORCA.py
#
# Controls for Hamamatsu ORCA-Flash4.0 V2 (C11440-22CU)
#
# Created: 20 September 2017
# Author : Z Marin
#
#
# Copyright David Baddeley, 2012
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
################

from .HamamatsuDCAM import *
from PYME.Acquire import eventLog
from PYME.Acquire.Hardware.Camera import MultiviewCameraMixin, CameraMapMixin
import logging
logger = logging.getLogger(__name__)

# C11440-22CU DCAM_IDPROP variables
# {'SUBARRAY VSIZE': 4202816, 'BUFFER TOP OFFSET BYTES': 4326224, 'IMAGE
# DETECTOR PIXEL NUM VERT': 4327488, 'INTERNAL FRAME INTERVAL': 4208672,
# 'OUTPUT TRIGGER ACTIVE[0]': 1835312, 'TRIGGER TIMES': 1049152,  'READOUT
# SPEED': 4194576, 'TIMING CYCLIC TRIGGER PERIOD': 4206624,  'TIMING INVALID
# EXPOSURE PERIOD': 4206704, 'INTERNAL LINE INTERVAL': 4208720,  'TRIGGER
# MODE': 1049104, 'IMAGE ROWBYTES': 4325936, 'BUFFER ROWBYTES': 4326192,
# 'IMAGE DETECTOR PIXEL NUM HORZ': 4327472,  'TIMING GLOBAL EXPOSURE DELAY':
# 4206736, 'OUTPUT TRIGGER PERIOD[0]': 1835344,  'OUTPUT TRIGGER PRE HSYNC
# COUNT': 1835408, 'BUFFER PIXEL TYPE': 4326240,  'OUTPUT TRIGGER POLARITY[
# 0]': 1835296, 'RECORD FIXED BYTES PER FILE': 4326416,  'BIT PER CHANNEL':
# 4325680, 'SUBARRAY HPOS': 4202768, 'BINNING': 4198672,  'INTERNAL LINE
# SPEED': 4208704, 'OUTPUT TRIGGER KIND[0]': 1835360,  'IMAGE DETECTOR PIXEL
# HEIGHT': 4327456, 'FRAME STAMP PRODUCER': 4262432,  'SUBARRAY VPOS':
# 4202800, 'IMAGE DETECTOR PIXEL WIDTH': 4327440,  'NUMBER OF OUTPUT TRIGGER
# CONNECTOR': 1835024, 'TIMING MIN TRIGGER BLANKING': 4206640,  'READOUT
# DIRECTION': 4194608, 'IMAGE HEIGHT': 4325920,  'TIMING MIN TRIGGER
# INTERVAL': 4206672, 'SENSOR COOLER STATUS': 2097984,  'IMAGE WIDTH':
# 4325904, 'TRIGGER CONNECTOR': 1049136, 'RECORD FIXED BYTES PER SESSION':
# 4326432, 'TRIGGER ACTIVE': 1048864, 'INTERNAL FRAME RATE': 4208656,
# 'TIMING READOUT TIME': 4206608, 'SYSTEM ALIVE': 16711696,  'TRIGGER GLOBAL
# EXPOSURE': 2032384, 'TRIGGER SOURCE': 1048848, 'IMAGE TOP OFFSET BYTES':
# 4325968, 'IMAGE PIXEL TYPE': 4326000, 'BUFFER FRAMEBYTES': 4326208,
# 'COLORTYPE': 4325664, 'SUBARRAY MODE': 4202832, 'TIMING EXPOSURE': 4206688,
#   'TIME STAMP PRODUCER': 4262416, 'CONVERSION FACTOR COEFF': 16769040,
# 'EXPOSURE TIME': 2031888, 'SUBARRAY HSIZE': 4202784,  'TRIGGER POLARITY':
# 1049120, 'DEFECT CORRECT MODE': 4653072, 'SENSOR MODE': 4194832,  'OUTPUT
# TRIGGER DELAY[0]': 1835328, 'OUTPUT TRIGGER SOURCE[0]': 1835280,  'RECORD
# FIXED BYTES PER FRAME': 4326448, 'SYNC READOUT SYSTEM BLANK': 1049232,
# 'TRIGGER DELAY': 1049184, 'CONVERSION FACTOR OFFSET': 16769056,  'IMAGE
# FRAMEBYTES': 4325952}

DCAMBUF_ATTACHKIND_FRAME = 0
DCAMPROP_MODE__OFF = 1
DCAMPROP_MODE__ON = 2

DCAMPROP_SENSORMODE__PROGRESSIVE = 12

DCAMPROP_TRIGGERSOURCE_INTERNAL = 1
DCAMPROP_TRIGGERSOURCE_EXTERNAL = 2
DCAMPROP_TRIGGERSOURCE_SOFTWARE = 3

DCAMPROP_OUTPUTTRIGGER_SOURCE__EXPOSURE = 1
DCAMPROP_OUTPUTTRIGGER_SOURCE__READOUTEND = 2
DCAMPROP_OUTPUTTRIGGER_SOURCE__VSYNC = 3
DCAMPROP_OUTPUTTRIGGER_SOURCE__HSYNC = 4
DCAMPROP_OUTPUTTRIGGER_SOURCE__TRIGGER = 6

DCAMPROP_OUTPUTTRIGGER_POLARITY__NEGATIVE = 1
DCAMPROP_OUTPUTTRIGGER_POLARITY__POSITIVE = 2

DCAMPROP_OUTPUTTRIGGER_ACTIVE__EDGE = 1
DCAMPROP_OUTPUTTRIGGER_ACTIVE__LEVEL = 2

DCAMPROP_OUTPUTTRIGGER_KIND__LOW = 1
DCAMPROP_OUTPUTTRIGGER_KIND__EXPOSURE = 2
DCAMPROP_OUTPUTTRIGGER_KIND__PROGRAMABLE = 3
DCAMPROP_OUTPUTTRIGGER_KIND__TRIGGER_READY = 4
DCAMPROP_OUTPUTTRIGGER_KIND__HIGH = 5
DCAMPROP_OUTPUTTRIGGER_KIND__ANYROWEXPOSURE = 6

DCAMCAP_START_SEQUENCE = ctypes.c_int32(int("-1",0))
#DCAMCAP_START_SNAP = ctypes.c_int32(int("0",0))

class DCAMZeroBufferedException(Exception):
    pass


class HamamatsuORCA(HamamatsuDCAM, CameraMapMixin):
    numpy_frames = 1

    def __init__(self, camNum):
        HamamatsuDCAM.__init__(self, camNum)

        self.waitopen = DCAMWAIT_OPEN()
        self.waitstart = DCAMWAIT_START()
        self.initialized = False
        self.bs = 0
        self.nReadOut = 0

        self._n_frames_leftover = 0

        self._aq_active = False

        self._last_framestamp = None
        self._last_camerastamp = None
        self._last_timestamp = None
        
        self._frameRate = 0

        # initialize other properties needed
        self.external_shutter = None

    def Init(self):
        logger.debug('Initializing Hamamatsu Orca')
        HamamatsuDCAM.Init(self)
        if self.camNum < camReg.maxCameras:
            # Create a wait handle
            self.waitopen.size = ctypes.sizeof(self.waitopen)
            self.waitopen.hdcam = self.handle
            self.checkStatus(dcam.dcamwait_open(ctypes.byref(self.waitopen)),
                             "dcamwait_open")
            self.waitstart.size = ctypes.sizeof(self.waitstart)
            self.waitstart.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY
            self.waitstart.timeout = DCAMWAIT_TIMEOUT_INFINITE
            self.setDefectCorrectMode(False)
            self.enable_cooling(True)
            self._mode = self.MODE_CONTINUOUS
            #self._mode = self.MODE_SINGLE_SHOT
            self.initialized = True
            logger.debug('Hamamatsu Orca initialized')


    
    def GetIntegTime(self):
        return self.getCamPropValue('EXPOSURE TIME')

    def setDefectCorrectMode(self, on=False):
        """
        Do we want Hamamatsu's hot pixel correction?

        Parameters
        ----------
        on : bool
            True if we want Hamamatsu's hot pixel correction.

        Returns
        -------
        None
        """
        onoff = 2.0 if on else 1.0
        self.setCamPropValue('DEFECT CORRECT MODE', onoff)

    def enable_cooling(self, on=True):
        #OFF =1 , ON = 2

        onoff = 2.0 if on else 1.0
        try:
            self.setCamPropValue('SENSOR COOLER', onoff)
        except DCAMException as e:
            # api v20.10.641, BT FUSION does not have SENSOR COOLER property
            # don't worry about it if cooling is already on 
            status = self.getCamPropValue('SENSOR COOLER STATUS')
            if status != onoff:
                raise e

        
    def GetAcquisitionMode(self):
        # Support both continuous and single shot modes
        #return self.MODE_CONTINUOUS
        return self._mode

    def SetAcquisitionMode(self, mode):
        if mode in [self.MODE_CONTINUOUS, self.MODE_SOFTWARE_TRIGGER, 
                    self.MODE_SINGLE_SHOT, self.MODE_HARDWARE_START_TRIGGER]:
            self._mode = mode
        else:
            raise RuntimeError('Mode %d not supported' % mode)


    def FireSoftwareTrigger(self):
        self.checkStatus(dcam.dcamcap_firetrigger(self.handle, 0), 'dcamcap_firetrigger')

    def StartExposure(self):
        self.nReadOut = 0
        self._n_frames_leftover = 0
        self._frameRate = self.getCamPropValue('INTERNAL FRAME RATE')
        # From parent
        HamamatsuDCAM.StartExposure(self)

        # Allocate buffers (2 seconds of buffers)
        self.bs = int(max(int(2.0*self._frameRate), 1))
        self.checkStatus(dcam.dcambuf_alloc(self.handle, ctypes.c_int32(
            self.bs)),
                         "dcambuf_alloc")

        self._image_frame_bytes = int(self.getCamPropValue('IMAGE FRAMEBYTES'))

        if self._mode == self.MODE_SOFTWARE_TRIGGER:
            self.setCamPropValue('TRIGGER SOURCE', DCAMPROP_TRIGGERSOURCE_SOFTWARE)
            self.checkStatus(dcam.dcamcap_start(self.handle,
                                            DCAMCAP_START_SEQUENCE),
                         "dcamcap_start")

        elif self._mode == self.MODE_CONTINUOUS:
            # Continuous mode, internal trigger
            self.setCamPropValue('TRIGGER SOURCE', DCAMPROP_TRIGGERSOURCE_INTERNAL)
            self.checkStatus(dcam.dcamcap_start(self.handle,
                                            DCAMCAP_START_SEQUENCE),
                         "dcamcap_start")

        elif self._mode == self.MODE_SINGLE_SHOT:
            # Spoofed single shot mode, using the software trigger
            # NOTE: this should no longer be needed when we add software trigger support to z-stepping etc ...
            self.setCamPropValue('TRIGGER SOURCE', DCAMPROP_TRIGGERSOURCE_SOFTWARE)
            self.checkStatus(dcam.dcamcap_start(self.handle,
                                            DCAMCAP_START_SEQUENCE),
                         "dcamcap_start")
            self.FireSoftwareTrigger()
        
        elif self._mode == self.MODE_HARDWARE_START_TRIGGER:
            self._set_trigger_start_mode()
            self.checkStatus(dcam.dcamcap_start(self.handle,
                                                DCAMCAP_START_SEQUENCE),
                                                "dcamcap_start")

        eventLog.logEvent('StartAq', '')

        # Start the capture
        #print str(self.getCamPropValue('SENSOR MODE'))
        self._aq_active = True
        return 0

    def SetROI(self, x1, y1, x2, y2):
        logger.debug('Setting ROI: x0 %3.1f, y0 %3.1f, w %3.1f, h %3.1f' %(x1, y1, x2-x1, y2-y1))

        #hamamatsu only supports ROI sizes (and positions) which are multiples of 4
        x1 = 4*np.floor(x1/4)
        y1 = 4*np.floor(y1/4)
        w = 4*np.floor((x2-x1)/4)
        h = 4*np.floor((y2-y1)/4)
        self.setCamPropValue('SUBARRAY MODE', DCAMPROP_MODE__OFF)
        self.setCamPropValue('SUBARRAY HPOS', x1)
        self.setCamPropValue('SUBARRAY HSIZE', w)
        self.setCamPropValue('SUBARRAY VPOS', y1)
        self.setCamPropValue('SUBARRAY VSIZE', h)
        self.setCamPropValue('SUBARRAY MODE', DCAMPROP_MODE__ON)

        logger.debug('ROI set: x0 %3.1f, y0 %3.1f, w %3.1f, h %3.1f' % (x1, y1, w, h))

    def GetROI(self):
        x1 = int(self.getCamPropValue('SUBARRAY HPOS'))
        y1 = int(self.getCamPropValue('SUBARRAY VPOS'))
        
        x2 = x1 + int(self.getCamPropValue('SUBARRAY HSIZE'))
        y2 = y1 + int(self.getCamPropValue('SUBARRAY VSIZE'))
        
        return x1, y1, x2, y2

    def StopAq(self):
        self._aq_active = False
        # Stop the capture
        self.checkStatus(dcam.dcamcap_stop(self.handle), "dcamcap_stop")

        # Free the buffers
        self.checkStatus(dcam.dcambuf_release(self.handle,
                                              DCAMBUF_ATTACHKIND_FRAME),
                         "dcambuf_release")

    def ExtractColor(self, chSlice, mode):
        # Ask the camera what frames it has available
        tInfo = DCAMCAP_TRANSFERINFO()
        tInfo.size = ctypes.sizeof(tInfo)
        tInfo.transferkind = ctypes.c_int32(0)
        self.checkStatus(dcam.dcamcap_transferinfo(self.handle,
                                                   ctypes.addressof(tInfo)),
                         "dcamcap_transferinfo")

        #Setup frame struct to tell the camera what frame we want
        frame = DCAMBUF_FRAME()
        frame.size = ctypes.sizeof(frame)
        #frame.iFrame = tInfo.nNewestFrameIndex  # Latest frame. This may need to be handled
        # differently.
        #frame.iFrame = ctypes.c_int32(self.nReadOut)

        # infer the frame number of the oldest image
        nframes_buffered = (tInfo.nFrameCount - self.nReadOut)

        next_frame = (tInfo.nNewestFrameIndex - nframes_buffered + 1)

        # Hamamatsu maintains a circular buffer, wrap around the end of the buffer if needed
        if (next_frame < 0):
            next_frame = self.bs + next_frame

        #do some sanity checks on the things we've calculated and print some stuff if they look weird
        if (nframes_buffered <1) or (tInfo.nNewestFrameIndex <0) or (next_frame < 0):
            print(nframes_buffered, next_frame, tInfo.nNewestFrameIndex)

        # if we've somehow got here but our calculations indicate that there are no new frames in the buffer, raise an
        # exception
        if nframes_buffered < 1:
            print(self._last_timestamp, self._last_framestamp, self._last_camerastamp)
            raise DCAMZeroBufferedException()

        #tell DCAM which fram we want from the buffer
        frame.iFrame = next_frame

        #lock / take ownership of the memory belonging to that frame
        self.checkStatus(dcam.dcambuf_lockframe(self.handle,
                                                ctypes.addressof(frame)),
                         "dcambuf_lockframe")

        # uint_16 for this camera only DCAM_IDPROP_BITSPERCHANNEL
        #print str(self.getCamPropValue('BUFFER FRAMEBYTES'))

        #copy into our local buffer
        ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_void_p(frame.buf),
            self._image_frame_bytes)

        self.nReadOut += 1
        self._n_frames_leftover = nframes_buffered -1

        self._last_camerastamp = frame.camerastamp
        self._last_framestamp = frame.framestamp
        self._last_timestamp = (frame.timestamp_sec, frame.timestamp_usec)


    def GetNumImsBuffered(self):
        tInfo = DCAMCAP_TRANSFERINFO()
        tInfo.size = ctypes.sizeof(tInfo)
        tInfo.transferkind = ctypes.c_int32(0)
        self.checkStatus(dcam.dcamcap_transferinfo(self.handle,
                                                    ctypes.addressof(tInfo)),
                         "dcamcap_transferinfo")
        ret = abs(int(tInfo.nFrameCount) - self.nReadOut)
        return ret

    def ExpReady(self):
        if not self._aq_active:
            return False

        if self._n_frames_leftover > 1:
            # if we somehow skipped events (i.e more than one frame came in between our last event and when we checked
            # buffer levels in extractColour), short circuit and read out the buffered frames. Stop at 1 leftover frame
            # as this possibly has an event waiting for it. It would appear that the DCAM api manages events by setting
            # flags (i.e. we get one event if any number of frames arrive between the last event we waited for and our
            # next wait call).
            return True
        try:
            # wait for a frame event - fired when the camera has a new frame. It would appear that one event gets fired
            # regardless of the number of frames that arrive between our last wait call and the this one.
            self.checkStatus(dcam.dcamwait_start(self.waitopen.hdcamwait, ctypes.byref(self.waitstart)),
                             "dcamwait_start")
        except DCAMException as e:
            logger.error(str(e))
            return False

        return self.waitstart.eventhappened == int(DCAMWAIT_CAPEVENT_FRAMEREADY.value)
        #return self.GetNumImsBuffered() > 0

    def GetBufferSize(self):
        return self.bs

    def SetIntegTime(self, intTime):
        [lb, ub] = self.getCamPropRange('EXPOSURE TIME')
        newTime = np.clip(intTime, lb, ub)
        print(str(newTime))
        self.setCamPropValue('EXPOSURE TIME', newTime)

    def GetCCDWidth(self):
        return int(self.getCamPropValue('IMAGE DETECTOR PIXEL NUM HORZ'))

    def GetCCDHeight(self):
        return int(self.getCamPropValue('IMAGE DETECTOR PIXEL NUM VERT'))

    def GetPicWidth(self):
        return int(self.getCamPropValue('IMAGE WIDTH'))

    def GetPicHeight(self):
        return int(self.getCamPropValue('IMAGE HEIGHT'))

    def CamReady(self):
        return self.initialized
    
    #@property
    #def noise_properties(self):
    #    return self.noiseProps
    
    def GetCCDTemp(self):
        # FIXME - actually read the CCD temperature
        return 0
    
    def GetFPS(self):
        return self._frameRate

    def GetCycleTime(self):
        """
        Get camera cycle time (1/fps) in seconds (float)

        Returns
        -------
        float
            Camera cycle time (seconds)
        """
        #FIXME - raise NotImplemeted?
    
        if self._frameRate > 0:
            return 1.0 / self._frameRate
    
        return 0.0

    def GetName(self):
        return "Hamamatsu ORCA Flash 4.0"

    def GenStartMetadata(self, mdh):
        HamamatsuDCAM.GenStartMetadata(self, mdh)
        if self.active:
            self.fill_camera_map_metadata(mdh)
            mdh.setEntry('Camera.ADOffset', self.noise_properties['ADOffset'])

    def SetShutter(self, mode):
        """
        This is a shim to use an external shutter, if ~assigned to the ORCA
        Parameters
        ----------
        mode : bool
            True (1) if open

        Returns
        -------
        None
        """
        if self.external_shutter is not None:
            self.external_shutter.SetShutter(mode)
    
    def SetOutputTrigger(self, mode, delay=0, width=0.0001, positive=True):
        """
        Set output trigger of the camera. For now, only sets output trigger 0, even if
        the camera supports multiple output triggers. 
        
        TODO: have a look at Andor and PCO SDKs to see if this is sufficiently similar in those, and potentially move/adapt to main camera spec.

        Parameters
        ----------
        mode : str
            Currently supported modes include:
                low: sets output trigger to Low, i.e. TTL zero. Can be useful 
                    for synchronization if one wants to change output trigger 
                    during an acquisition protocol. Ignores pulse delay and 
                    width parameters
                readout start: sets output trigger to 'Vsync', which gives the 
                    TTL of the specified pulse width output after the specified
                    delay from the start of the sensor readout
                readout end: sets output trigger to readout end, which gives
                    the TTL of the specified pulse width output after the
                    specified delay from the end of the sensor readout
        delay : float, optional
            delay after trigger event, in seconds, to emit TTL high (assuming 
            posiive polarity), by default 0 s.
        width : float, optional
            TTL high pulse width, in seconds, by default 0.0001 s, or 0.1 ms
        positive : bool, optional
            Sets polarity of the output trigger to positive (True) or negative
            (False). True, by default.
        
        """
        if positive:
            self.setCamPropValue('OUTPUT TRIGGER POLARITY[0]', 
                                DCAMPROP_OUTPUTTRIGGER_POLARITY__POSITIVE)
        else:
            self.setCamPropValue('OUTPUT TRIGGER POLARITY[0]', 
                                DCAMPROP_OUTPUTTRIGGER_POLARITY__NEGATIVE)
        
        if mode == 'low':
            self.setCamPropValue('OUTPUT TRIGGER KIND[0]', 
                                 DCAMPROP_OUTPUTTRIGGER_KIND__LOW)
            return  # return early, no need to set delay and width
        
        # in case the camera is running, set the pulse parameters before 
        # changing the trigger source
        # self.setCamPropValue('OUTPUT TRIGGER ACTIVE[0]', DCAMPROP_OUTPUTTRIGGER_ACTIVE__EDGE)  # not writable
        self.setCamPropValue('OUTPUT TRIGGER DELAY[0]', delay)  # [s], only relevant with 'EDGE'
        self.setCamPropValue('OUTPUT TRIGGER PERIOD[0]', width)  # [s], width of pulse, only relevant with 'EDGE'
        if mode == 'readout start':
            self.setCamPropValue('OUTPUT TRIGGER KIND[0]',
                                 DCAMPROP_OUTPUTTRIGGER_KIND__PROGRAMABLE)
            self.setCamPropValue('OUTPUT TRIGGER SOURCE[0]',
                                 DCAMPROP_OUTPUTTRIGGER_SOURCE__VSYNC)
        elif mode == 'readout end':
            self.setCamPropValue('OUTPUT TRIGGER KIND[0]',
                                 DCAMPROP_OUTPUTTRIGGER_KIND__PROGRAMABLE)
            self.setCamPropValue('OUTPUT TRIGGER SOURCE[0]',
                                 DCAMPROP_OUTPUTTRIGGER_SOURCE__READOUTEND)
        else:
            raise RuntimeError('Unsupported output trigger mode: %s' % mode)
    
    def _set_trigger_start_mode(self):
        """Use to start internally-timed acquisition starting on an external
        hardware trigger (currently hardcoded to edge)

        Notes
        -----
        set DCAMPROP_TRIGGERSOURCE__EXTERNAL as DCAM_IDPROP_TRIGGERSOUCE 
        and DCAMPROP_TRIGGER_MODE__START as DCAM_IDPROP_TRIGGER_MODE. The
        camera changes to internal mode when the camera receives the trigger. 
        The DCAM_IDPROP_TRIGGERSOURCE property will be 
        DCAMPROP_TRIGGERSOURCE__INTERNAL automatically.
        """
        try:
            self.setCamPropValue('TRIGGER ACTIVE', DCAMPROP_TRIGGERACTIVE__EDGE)
        except:
            # Sometimes TRIGGER ACTIVE is not writable
            pass
        self.setCamPropValue('TRIGGER SOURCE', DCAMPROP_TRIGGERSOURCE_EXTERNAL)
        self.setCamPropValue('TRIGGER MODE', DCAMPROP_TRIGGER_MODE__START)
    
    def _get_global_exposure_delay(self):
        """How long from the beginning of exposure does it take before
        all lines in active ROI are being exposed (global exposure)

        Returns
        -------
        delay: float
            Global exposure delay, in [s]

        """
        return self.getCamPropValue('TIMING GLOBAL EXPOSURE DELAY')

    def Shutdown(self):
        # if self.initialized:
            # self.checkStatus(dcam.dcamwait_close(self.waitopen.hdcamwait),
            #                "dcamwait_close")
        HamamatsuDCAM.Shutdown(self)

class Fusion(HamamatsuORCA):
    """
    Orca Fusion is functionally the same as the Flash, however uses multiple gain modes.
    TODO - check Flash return/fail on READOUT SPEED property so we can catch/return 'fixed'
    and not necessarily introduce an extra class
    """
    _gain_modes = {
        1:'Ultra-quiet',
        2:'Standard',
        3:'Fast'
    }

    @property
    def _gain_mode(self):
        return self._gain_modes[int(self.getCamPropValue('READOUT SPEED'))]


class MultiviewOrca(MultiviewCameraMixin, HamamatsuORCA):
    def __init__(self, camNum, multiview_info):
        HamamatsuORCA.__init__(self, camNum)
        # default to the whole chip
        default_roi = dict(xi=0, xf=2048, yi=0, yf=2048)
        MultiviewCameraMixin.__init__(self, multiview_info, default_roi, HamamatsuORCA)


#TODO - replace MultiviewCameraMixin with a Multiview wrapper so that we don't need to have explicit multiview versions of all cameras.
class MultiviewFusion(MultiviewCameraMixin, Fusion):
    def __init__(self, camNum, multiview_info):
        Fusion.__init__(self, camNum)
        # default to the whole chip
        default_roi = dict(xi=0, xf=2304, yi=0, yf=2304)
        MultiviewCameraMixin.__init__(self, multiview_info, default_roi, Fusion)
