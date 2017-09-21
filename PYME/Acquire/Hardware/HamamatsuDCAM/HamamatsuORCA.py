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
from PYME.Acquire.Hardware.sCMOSCamera import sCMOSCamera
from fftw3f import create_aligned_array


# C11440-22CU DCAM_IDPROP variables


class HamamatsuORCA(HamamatsuDCAM, sCMOSCamera):

    def __init__(self):
        HamamatsuDCAM.__init__(self)

        # initialize other properties needed

    def InitBuffers(self):
        self._flush()

        #bufSize = self.ImageSizeBytes.getValue()
        #vRed = int(self.SensorHeight.getValue() / self.AOIHeight.getValue())
        #self.nBuffers = vRed * self.defBuffers

        if not self.contMode:
            self.nBuffers = 5

        # print bufSize
        for i in range(self.nBuffers):
            # buf = np.empty(bufSize, 'uint8')
            buf = create_aligned_array(bufSize, 'uint8')
            self._queueBuffer(buf)

        self.doPoll = True

    def _flush(self):
        # Turn off camera polling
        self.doPoll = False

        # flush camera buffers

        # flush local buffers
        sCMOSCamera._flush(self)

        # flush camera buffers again

    def _queueBuffers(self):
        """
        Grab camera buffers and stash them in queuedBuffers, moving the data
        first through buffersToQueue. This function is too camera-specific
        for any general code in the class.
        nQueued += 1
        """
        pass

    def _pollBuffer(self):
        """
        Grabbed the queuedBuffers off the camera and stash them in fullBuffers.
        This function is too camera-specific for any general code in the class.
        nFull += 1
        """
        pass

    def ExtractColor(self, chSlice, mode):
        """
        Pulls the oldest frame from the camera buffer and copies it into
        memory we provide. Note that the function signature and parameters are
        a legacy of very old code written for a colour camera with a bayer mask.
        This function is too camera-specific for any general code in the class.

        Parameters
        ----------
        chSlice : `~numpy.ndarray`
            The array we want to put the data into
        mode : int
            Previously specified how to deal with the Bayer mask.

        Returns
        -------
        None
        """

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