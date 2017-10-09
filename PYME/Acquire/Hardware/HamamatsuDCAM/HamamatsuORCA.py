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

noiseProperties = {
1823 : {
        'ReadNoise' : 109.8,
        'ElectronsPerCount' : 27.32,
        'NGainStages' : 536,
        'ADOffset' : 971,
        'DefaultEMGain' : 150,
        'SaturationThreshold' : (2**14 -1)
        }
}


class HamamatsuORCA(HamamatsuDCAM):

    def __init__(self):
        HamamatsuDCAM.__init__(self)

        #self.noiseProps = noiseProperties[self.SerialNumber]

        # initialize other properties needed

    def StartExposure(self):
        """
        Starts an acquisition.

        Returns
        -------
        int
            Success (0) or failure (-1) of initialization.
        """

        HamamatsuDCAM.StartExposure(self)

        # start the acquisition

        return 0

    def StopAq(self):
        """
        Stops acquiring.

        Returns
        -------
        None
        """
        pass

    def ExtractColor(self, chSlice, mode):
        # DCAM lockframe
        # ctypes memcpy AndorNeo 251
        frame = DCAMBUF_FRAME()
        frame.size = ctypes.sizeof(frame)
        frame.iFrame = -1 # Latest frame. This may need to be handled
        # differently.
        self.checkStatus(dcam.dcambuf_lockframe(self.handle,
                                                ctypes.addressof(frame)),
                         "dcambuf_lockframe")
        #ctypes.cdll.msvcrt.memcpy(chSlice.ctypes.data_as(ctypes.POINTER(
        # ctypes.c_uint8)), buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), chSlice.nbytes)

        pass


    def SetBurst(self, burstSize):
        """
        Some support for burst mode on the Neo and Zyla. Is not called from
        the GUI, and can be considered experimental and non-essential.

        Parameters
        ----------
        burtSize : int
            Number of frames to acquire in burst mode.

        Returns
        -------
        None
        """
        pass