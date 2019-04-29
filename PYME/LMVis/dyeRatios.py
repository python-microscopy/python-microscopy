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
"""
Determine the splitting ratio for a given dye. Uses the following (in order of increasing precedence)

1) Hard coded values from this file
2) Values from the `Splitter.Ratios` metadata key if set in the acquisition software
3) Values contained in a json formatted file referenced by the PYME.config parameter `SplitterRatioDatabase`

If supplied, the SplitterRatioDatabase should be the filename of a json file with a nested dictionary following the
format given below:

.. code-block:: json

    {
        "Microsocope_ID_1" : {
            "DICHROIC_1" : { "Dye1" : ratio1, "Dye2" : ratio2, "Dye3" : ratio3},
            "DICHROIC_2" : { "Dye1" : ratio1, "Dye2" : ratio2},
        },
        "Microscope_ID_2" : {
            "DICHROIC_1" : { "Dye1" : ratio1, "Dye2" : ratio2, "Dye3" : ratio3},
        },
    }

"""

from PYME import config
import os
import json

import logging
logger = logging.getLogger(__name__)

ratios = {'A647':0.85, 'A680':0.87, 'A750': 0.11, 'A700': 0.3, 'CF770': 0.11}
PRIRatios = {'A680':0.7, 'A750': 0.5}

dichr_ratios = {
    # standard splitter for 680/750
    'FF700-Di01' : {'A647':0.3, 'A680':0.87,'A700':0.7, 'A750':.9},
    # older Auckland based splitter
    'FF741-Di01': ratios,
    # Chroma splitter for 647/700 with Semrock SP01-785R in long arm
    'T710LPXXR-785R' : {'A647': 0.79, 'ATTO655': 0.74, 'Atto655': 0.74, 'AT655': 0.74, 'AT700': 0.22, 'ATTO700':0.22, 'A700':0.43},
    # Chroma splitter for 647/700 WITHOUT Semrock SP01-785R in long arm
    'T710LPXXR' : {'A700' : 0.39, 'A647' : 0.73}
    }

ratios_by_machine = {
    'default' : dichr_ratios,
}

ratio_filename = config.get('SplitterRatioDatabase', None)
if (not ratio_filename is None) and os.path.exists(ratio_filename):
    with open(ratio_filename, 'r') as f:
        ratios_by_machine.update(json.load(f))

def get_ratios(dichroic=None, acquisition_machine='default'):
    if dichroic is None:
        return ratios
    
    try:
        return ratios_by_machine[acquisition_machine][dichroic]
    except KeyError:
        logger.warning('Unknown dichroic "%s"' % dichroic)
        return {}
    

def getRatio(dye, mdh=None):
    #load defaults
    split_ratios = get_ratios(mdh.getOrDefault('Splitter.Dichroic', None)).copy()
        
    #read from acquisition metadata if present
    if 'Splitter.Ratios' in mdh.getEntryNames():
        split_ratios.update(mdh['Splitter.Ratios'])

    #finally try to update from database if we have both a machine name and a splitter name
    try:
        split_ratios.update(ratios_by_machine[mdh['Splitter.Dichroic']][mdh['MicroscopeName']])
    except (KeyError, AttributeError):
        pass
        
    try:
        #look up the dye
        return split_ratios[dye]
    
    except KeyError:
        # this path is likely never reached
        if dye in ratios.keys():
            if 'Analysis.FitModule' in mdh.getEntryNames() and mdh['Analysis.FitModule'].startswith('PRInterpFit'):
                return PRIRatios[dye]
            if 'Splitter.TransmittedPathPosition' in mdh.getEntryNames() and mdh.getEntry('Splitter.TransmittedPathPosition') == 'Top':
                return 1 - ratios[dye]
            else:
                return ratios[dye]
        else:
            return None # default action
