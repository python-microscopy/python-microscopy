#!/usr/bin/python

##################
# sensicamMetadata.py
#
# Copyright David Baddeley, 2009
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
##################

from PYME.IO import MetaDataHandler

class sensicamMD:
    def __init__(self, cam):
        self.cam = cam
        self.noiseProps = {
            'ReadNoise' : 4,
            'ElectronsPerCount' : 2,
            'NGainStages' : 1,
            'ADOffset' : 20,
            'SaturationThreshold' : (2**12 -1)
            }

        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

    def GenStartMetadata(self, mdh):
        self.cam.GetStatus()

        mdh.setEntry('Camera.Name', 'Sensicam QE')

        mdh.setEntry('Camera.IntegrationTime', self.cam.GetIntegTime())
        mdh.setEntry('Camera.CycleTime', self.cam.GetCycleTime())
        mdh.setEntry('Camera.EMGain', 1)

        mdh.setEntry('Camera.ROIPosX', self.cam.GetROIX1())
        mdh.setEntry('Camera.ROIPosY',  self.cam.GetROIY1())
        mdh.setEntry('Camera.ROIWidth', self.cam.GetROIX2() - self.cam.GetROIX1())
        mdh.setEntry('Camera.ROIHeight',  self.cam.GetROIY2() - self.cam.GetROIY1())
        mdh.setEntry('Camera.StartCCDTemp',  self.cam.GetCCDTemp())

        mdh.setEntry('Camera.BinningX', self.cam.GetHorizBin())
        mdh.setEntry('Camera.BinningY',  self.cam.GetVertBin())

        #these should really be read from a configuration file
        #hard code them here until I get around to it
        #current values are for the low light mode
        mdh.setEntry('Camera.ReadNoise', 4)
        mdh.setEntry('Camera.NoiseFactor', 1)
        mdh.setEntry('Camera.ElectronsPerCount', 2)

        mdh.setEntry('Camera.TrueEMGain', 1)
