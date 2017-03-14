#!/usr/bin/python

##################
# h5ToTiffSeries.py
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

import tables
from PIL import Image
import sys
import os

if __name__ == '__main__':
    if not (len(sys.argv) == 3):
        raise RuntimeError('Usage: h5ToTiffSeries infile outdir')
    
    inFile = sys.argv[1]
    outDir = sys.argv[2]
    
    h5f = tables.open_file(inFile)
    
    nSlices = h5f.root.ImageData.shape[0]
    
    if os.path.exists(outDir):
        raise RuntimeError('Destination already exists')
    
    os.makedirs(outDir)
    
    
    for i in range(nSlices):
        Image.fromarray(h5f.root.ImageData[i, :,:].squeeze(), 'I;16').save(os.path.join(outDir, 'frame_%03d.tif'%i))    



