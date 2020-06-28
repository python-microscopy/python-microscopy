#!/usr/bin/python

##################
# IJLut.py
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
# import pylab
import matplotlib.cm
import numpy
import sys
import os

if __name__ == "__main__":

    if not len(sys.argv) == 2:
        raise RuntimeError('expected a directory to save the luts to')

    outDir = sys.argv[1]

    cmapnames = matplotlib.cm.cmapnames

    for cmn in cmapnames:
        c = (255*matplotlib.cm.__dict__[cmn](numpy.arange(256)))[:,:3].astype('uint8')

        f = open(os.path.join(outDir, '%s.lut' % cmn), 'wb')
        c.T.tofile(f)
        f.close()
