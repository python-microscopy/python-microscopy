#!/usr/bin/python

##################
# kdfToImage.py
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

from PIL import Image
import read_kdf
import sys

if not (len(sys.argv) == 3):
    raise RuntimeError('Usage: kdfToImage infile outfile')

inFile = sys.argv[1]
outFile = sys.argv[2]

im = read_kdf.ReadKdfData(inFile).squeeze()

mode = ''

if (im.dtype.__str__() == 'uint16'):
    mode = 'I;16'
elif (im.dtype.__str__() == 'float32'):
    mode = 'F'
else:
    raise RuntimeError('Error data type <%s> not supported') % im.dtype

Image.fromarray(im, mode).save(outFile)
