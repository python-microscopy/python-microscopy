#!/usr/bin/python

##################
# compHdf.py
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
import sys

if __name__ == '__main__':  
    if not (len(sys.argv) in [2,3]):
        raise RuntimeError('Usage: kdfToImage infile [outfile]')
    
    inFile = sys.argv[1]
    
    if len(sys.argv) == 2:
        outFile = inFile.split('.')[0] + '_c.h5'
    else:
        outFile = sys.argv[2]
    
    inF = tables.open_file(inFile)
    outF = tables.open_file(outFile, 'w')
