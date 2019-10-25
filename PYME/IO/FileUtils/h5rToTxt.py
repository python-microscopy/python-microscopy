#!/usr/bin/python

##################
# h5rToTxt.py
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

from PYME.IO import tabular
import os
import sys


def convertFile(inFile, outFile):
    ds = tabular.H5RSource(inFile)

    nRecords = len(ds[ds.keys()[0]])

    of = open(outFile, 'w')

    of.write('#' + '\t'.join(['%s' % k for k in ds._keys]) + '\n')

    for row in zip(*[ds[k] for k in ds._keys]):
        of.write('\t'.join(['%e' % c for c in row]) + '\n')

    of.close()


def saveFilter(ds, outFile, keys = None):
    if keys is None:
        keys = ds.keys()

    #nRecords = len(ds[keys[0]])

    of = open(outFile, 'w')

    of.write('#' + '\t'.join(['%s' % k for k in keys]) + '\n')

    for row in zip(*[ds[k] for k in keys]):
        of.write('\t'.join(['%e' % c for c in row]) + '\n')

    of.close()



if __name__ == '__main__':

    if (len(sys.argv) == 3):
        inFile = sys.argv[1]
        outFile = sys.argv[2]
    elif (len(sys.argv) == 2):
        inFile = sys.argv[1]
    else:
        raise RuntimeError('Usage: h5rToTxt.py infile [outFile]')

    if (len(sys.argv) == 2): #generate an output file name
        outFile = os.path.splitext(inFile)[0] + '.txt'

    if os.path.exists(outFile):
        print('Output file already exists - please remove')
    else:
        convertFile(inFile, outFile)
