#!/usr/bin/python

##################
# HDFDataSource.py
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

from PYME.IO.DataSources.FlatFieldDataSource import DataSource as FlatFieldDataSource


class DataSource(FlatFieldDataSource):
    moduleName = 'ElectronsPerSecondDataSource'
    def __init__(self, parentSource, mdh, flatfield=None, dark=None):
        """
        Create a datasource which wraps a series in units of ADU and returns frames in units of
        photoelectrons per second (e/s). parentSource should not be camera map corrected (dark-
        or flatfield-corrected).
        
        """
        if mdh.getOrDefault('Units', 'ADU') == 'e/s':  # TODO - rather than hard throw just don't further correct the parentSource
            raise RuntimeError('units are already e/s')
        
        FlatFieldDataSource.__init__(self, parentSource, mdh, flatfield, dark)
        
        self._adu_to_epers = self.mdh['Camera.ElectronsPerCount'] / self.mdh['Camera.TrueEMGain'] / self.mdh['Camera.IntegrationTime']

    def getSlice(self, ind):
        # flatfield getSlice will subtract the dark map and flatfield. 
        return FlatFieldDataSource.getSlice(self, ind) * self._adu_to_epers  # corrected ADU to photoelectrons/second
