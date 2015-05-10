#!/usr/bin/python

##################
# hdfWatcher.py
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

import sys
import time
import Pyro.core
import remFitHDF
import tables
import MetaData

if not len(sys.argv) == 4:
    raise RuntimeError('usage: hdfWatcher filename threshold poll_time')

fname = sys.argv[1]
thresh = float(sys.argv[2])
slTime = float(sys.argv[3])


curpos = 0

tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')

while 1:
    h5file = tables.openFile(fname)
    seriesName = h5file.filename
    l = h5file.root.ImageData.shape[0]
    #h5file.close()

    if l > curpos:
        for i in range(curpos, l):
            print(i)
            tq.postTask(remFitHDF.fitTask(seriesName,i, thresh, MetaData.TIRFDefault, 'LatGaussFitF', bgindices=range(max(i-10, 0),i), SNThreshold=True), queueName=seriesName)
            
    if 'EndTime' in h5file.root.MetaData._v_attrs:
        break

    h5file.close()
    curpos = l

    time.sleep(slTime)
