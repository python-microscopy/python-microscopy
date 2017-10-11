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
from fftw3f import create_aligned_array


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

DCAMCAP_START_SEQUENCE = ctypes.c_int32(int("-1",0))

noiseProperties = {
'S/N: 100233' : {
        'ReadNoise': 1.65,
        'ElectronsPerCount': 0.47,
        'NGainStages': 0,
        'ADOffset': 100,
        'DefaultEMGain': 1,
        'SaturationThreshold': (2**16 - 1)
        }
}


class HamamatsuORCA(HamamatsuDCAM):

    numpy_frames = 1

    def __init__(self, camNum):
        HamamatsuDCAM.__init__(self, camNum)

        self.noiseProps = {}
        self.waitopen = DCAMWAIT_OPEN()
        self.waitstart = DCAMWAIT_START()
        self.initialized = False
        self.bs = 0
        self.nReadOut = 0

        # initialize other properties needed

    def Init(self):
        HamamatsuDCAM.Init(self)
        if self.camNum < camReg.maxCameras:
            self.noiseProps = noiseProperties[self.SerialNumber]
            # Create a wait handle
            self.waitopen.size = ctypes.sizeof(self.waitopen)
            self.waitopen.hdcam = self.handle
            self.checkStatus(dcam.dcamwait_open(ctypes.byref(self.waitopen)),
                             "dcamwait_open")
            self.waitstart.size = ctypes.sizeof(self.waitstart)
            self.waitstart.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY
            self.waitstart.timeout = DCAMWAIT_TIMEOUT_INFINITE
            self.setDefectCorrectMode(False)
            self.initialized = True


    @property
    def _intTime(self):
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

    def StartExposure(self):
        self.nReadOut = 0
        self._frameRate = self.getCamPropValue('INTERNAL FRAME RATE')
        # From parent
        HamamatsuDCAM.StartExposure(self)

        # Allocate buffers (2 seconds of buffers)
        self.bs = int(max(int(2.0*self._frameRate), 1))
        self.checkStatus(dcam.dcambuf_alloc(self.handle, ctypes.c_int32(
            self.bs)),
                         "dcambuf_alloc")

        # Start the capture
        #print str(self.getCamPropValue('SENSOR MODE'))
        self.checkStatus(dcam.dcamcap_start(self.handle,
                                            DCAMCAP_START_SEQUENCE),
                         "dcamcap_start")

        return 0

    def SetROI(self, x1, y1, x2, y2):
        self.setCamPropValue('SUBARRAY HPOS', x1)
        self.setCamPropValue('SUBARRAY HSIZE', x2-x1)
        self.setCamPropValue('SUBARRAY VPOS', y1)
        self.setCamPropValue('SUBARRAY VSIZE', y2-y1)

        # If our ROI doesn't span the whole CCD, turn on subarray mode
        if x2-x1 == self.GetCCDWidth() and y2-y1 == self.GetCCDHeight():
            self.setCamPropValue('SUBARRAY MODE', DCAMPROP_MODE__OFF)
        else:
            self.setCamPropValue('SUBARRAY MODE', DCAMPROP_MODE__ON)

    def GetROIX1(self):
        return int(self.getCamPropValue('SUBARRAY HPOS'))

    def GetROIX2(self):
        return int(self.GetROIX1() + self.getCamPropValue('SUBARRAY HSIZE'))

    def GetROIY1(self):
        return int(self.getCamPropValue('SUBARRAY VPOS'))

    def GetROIY2(self):
        return int(self.GetROIY1() + self.getCamPropValue('SUBARRAY VSIZE'))

    def StopAq(self):
        # Stop the capture
        self.checkStatus(dcam.dcamcap_stop(self.handle), "dcamcap_stop")

        # Free the buffers
        self.checkStatus(dcam.dcambuf_release(self.handle,
                                              DCAMBUF_ATTACHKIND_FRAME),
                         "dcambuf_release")

    def ExtractColor(self, chSlice, mode):
        # DCAM lockframe
        # ctypes memcpy AndorNeo 251
        #print "Frame rate: " + str(self._frameRate)
        tInfo = DCAMCAP_TRANSFERINFO()
        tInfo.size = ctypes.sizeof(tInfo)
        tInfo.transferkind = ctypes.c_int32(0)
        self.checkStatus(dcam.dcamcap_transferinfo(self.handle,
                                                   ctypes.addressof(tInfo)),
                         "dcamcap_transferinfo")
        frame = DCAMBUF_FRAME()
        frame.size = ctypes.sizeof(frame)
        frame.iFrame = tInfo.nNewestFrameIndex  # Latest frame. This may need to be handled
        # differently.
        self.checkStatus(dcam.dcambuf_lockframe(self.handle,
                                                ctypes.addressof(frame)),
                         "dcambuf_lockframe")

        # uint_16 for this camera only DCAM_IDPROP_BITSPERCHANNEL
        #print str(self.getCamPropValue('BUFFER FRAMEBYTES'))
        ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)),
            frame.buf,
            int(self.getCamPropValue('IMAGE FRAMEBYTES')))
        self.nReadOut += 1

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
        self.checkStatus(dcam.dcamwait_start(self.waitopen.hdcamwait,
                                             ctypes.byref(self.waitstart)),
                         "dcamwait_start")
        return self.waitstart.eventhappened == int(
            DCAMWAIT_CAPEVENT_FRAMEREADY.value)
        #return self.GetNumImsBuffered() > 0

    def GetBufferSize(self):
        return self.bs

    def SetIntegTime(self, intTime):
        [lb, ub] = self.getCamPropRange('EXPOSURE TIME')
        newTime = np.clip(intTime, lb, ub)
        print str(newTime)
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

    def GetStatus(self):
        # This is to shut the command line the hell up.
        pass

    def GetReadNoise(self):
        return self.noiseProps['ReadNoise']

    def GetElectrPerCount(self):
        return self.noiseProps['ElectronsPerCount']

    def GetName(self):
        return "Hamamatsu ORCA Flash 4.0"

    def GenStartMetadata(self, mdh):
        HamamatsuDCAM.GenStartMetadata(self, mdh)
        if self.active:
            mdh.setEntry('Camera.ADOffset', self.noiseProps['ADOffset'])

    def Shutdown(self):
        # if self.initialized:
            # self.checkStatus(dcam.dcamwait_close(self.waitopen.hdcamwait),
            #                "dcamwait_close")
        HamamatsuDCAM.Shutdown(self)
