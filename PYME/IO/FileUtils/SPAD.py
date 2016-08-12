##################
# SPAD.py
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

import numpy as np

def loadRaw(filename):
    """Loads a series of frames from a spad output file. Despite the .txt extension
    the data files are in fact pure binary data. Frames are concatenated one after the
    other in the file. Frame size is assumed to be 32x32 pixels and the data type assumed
    to be unsigned short."""
    return np.fromfile(filename, 'uint16').reshape((32,32,-1), order='F')

def loadDark(filename):
    """Loads dark count information from a specified file. Note that this is
    referred to as flat-field information within the acquisition gui, but as it is
    subtracted in subsequent processing, this nomenclature is potentially misleading."""

    return np.fromfile(filename,sep=',').reshape((32,32), order='F')

def load(filename, tIntegration=0., tDead=0., nAccumulation=1., darkFile=None, clip=True):
    """load a series, and perform dead time and dark-count correction."""

    data = loadRaw(filename)

    if tDead > 0: #dead time correction
        data = data/(1.-(data/nAccumulation)*(tDead/(tIntegration*10.0)))

    if not darkFile is None: #we have a dark image - calibrate & subtract it
        dark = loadDark(darkFile)*(tIntegration/1.e8)*nAccumulation

        data = data - dark[:,:,None]

    if clip:
        data = np.clip(data,0,tIntegration*10.0*nAccumulation/tDead)

    return data
