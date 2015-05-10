#!/usr/bin/python

###############
# dyeRatios.py
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
'''Placeholder module for a more complicated storage of dye ratios, including
different splitter configurations.

currently just uses values for the default splitter config'''

ratios = {'A647':0.85, 'A680':0.87, 'A750': 0.11, 'A700': 0.3, 'CF770': 0.11}
PRIRatios = {'A680':0.7, 'A750': 0.5}

dichr_ratios = {'FF700-Di01': {'A647':0.3, 'A680':0.87,'A700':0.7, 'A750':.9}}

def getRatio(dye, mdh=None):
    if dye in ratios.keys():
        
        if 'Analysis.FitModule' in mdh.getEntryNames() and mdh['Analysis.FitModule'].startswith('PRInterpFit'):
            return PRIRatios[dye]
        if 'Splitter.Dichroic' in mdh.getEntryNames():
            dichroicName = mdh['Splitter.Dichroic']
            if dichroicName in dichr_ratios.keys():
                return dichr_ratios[dichroicName][dye]
                
        if 'Splitter.TransmittedPathPosition' in mdh.getEntryNames() and mdh.getEntry('Splitter.TransmittedPathPosition') == 'Top':
            return 1 - ratios[dye]
        else:
            return ratios[dye]
    else:
        return None