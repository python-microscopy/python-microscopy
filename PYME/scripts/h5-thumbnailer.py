#!/usr/bin/python

##################
# h5-thumbnailer.py
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
#import gnomevfs

import tables

#from PYME.LMVis import inpFilt
from scipy import minimum, maximum
from PIL import Image

from PYME.Analysis import MetaData
from PYME.IO import MetaDataHandler
from PYME.IO.DataSources import HDFDataSource

#import logging
#LOG_FILENAME = '/tmp/h5r-thumbnailer.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)

size = (200,200)


def generateThumbnail(inputFile, thumbSize):
    global size
    #logging.debug('Input File: %s\n' % inputFile)
    #logging.debug('Ouput File: %s\n' % outputFile)
    #logging.debug('Thumb Size: %s\n' % thumbSize)

    h5f = tables.open_file(inputFile)

    dataSource = HDFDataSource.DataSource(inputFile, None)

    md = MetaData.genMetaDataFromSourceAndMDH(dataSource, MetaDataHandler.HDFMDHandler(h5f))


    xsize = h5f.root.ImageData.shape[1]
    ysize = h5f.root.ImageData.shape[2]

    if xsize > ysize:
        zoom = float(thumbSize)/xsize
    else:
        zoom = float(thumbSize)/ysize

    size = (int(xsize*zoom), int(ysize*zoom))

    im = h5f.root.ImageData[min(md['EstimatedLaserOnFrameNo']+10,(h5f.root.ImageData.shape[0]-1)) , :,:].astype('f')

    im = im.T - min(md['Camera.ADOffset'], im.min())

    h5f.close()

    im = maximum(minimum(1*(255*im)/im.max(), 255), 0)

    return im.astype('uint8')

if __name__ == '__main__':
    import gnomevfs
    inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
    outputFile = sys.argv[2]
    thumbSize = int(sys.argv[3])

    im = generateThumbnail(inputFile, thumbSize)

    Image.fromarray(im).resize(size).save(outputFile, 'PNG')
