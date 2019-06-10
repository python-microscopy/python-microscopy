#!/usr/bin/python

###############
# __init__.py
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
"""Data sources are a modular method for reading data (i.e. input plugins)

Each DataSource should implement at least the following functions:

getSlice(ind)  
getSliceShape()
getNumSlices()

For more details see BaseDataSource

"""
import os

def getDataSourceForFilename(filename):
    if filename.startswith('QUEUE://'):
        from . import TQDataSource
        return TQDataSource.DataSource
    elif filename.startswith('PYME-CLUSTER://') or filename.startswith('pyme-cluster://'):
        from . import ClusterPZFDataSource
        return ClusterPZFDataSource.DataSource
    elif filename.startswith('http://') or filename.startswith('HTTP://'):
        from . import HTTPDataSource
        return HTTPDataSource.DataSource
    elif filename.endswith('.h5'):
        from . import HDFDataSource
        return HDFDataSource.DataSource
    #elif filename.endswith('.md'): #treat this as being an image series
    #    self._loadImageSeries(filename)
    elif os.path.splitext(filename)[1] in ['.tif', '.tif', '.lsm']: #try tiff
        from . import TiffDataSource
        return TiffDataSource.DataSource
    elif filename.endswith('.dcimg'):
        from . import DcimgDataSource
        return DcimgDataSource.DataSource
    else: #try bioformats
        from . import BioformatsDataSource
        return BioformatsDataSource.DataSource