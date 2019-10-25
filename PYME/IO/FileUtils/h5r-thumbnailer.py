#!/usr/bin/python

##################
# h5r-thumbnailer.py
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

#!/usr/bin/python

#import logging
#LOG_FILENAME = '/tmp/h5r-thumbnailer.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)

import sys
#import gnomevfs

from PYME.IO import tabular
from scipy import histogram2d, arange, minimum, concatenate, newaxis
from PIL import Image


#logging.debug('Input File: %s\n' % inputFile)
#logging.debug('Ouput File: %s\n' % outputFile)
#logging.debug('Thumb Size: %s\n' % thumbSize)

def generateThumbnail(inputFile, thumbSize):
    f1 = tabular.H5RSource(inputFile)

    threeD = False
    stack = False
    split = False

    #print f1.keys()

    if 'fitResults_Ag' in f1.keys():
        #if we used the splitter set up a mapping so we can filter on total amplitude and ratio
        f1_ = tabular.MappingFilter(f1, A='fitResults_Ag + fitResults_Ar', gFrac='fitResults_Ag/(fitResults_Ag + fitResults_Ar)')
        #f2 = inpFilt.resultsFilter(f1_, error_x=[0,30], A=[5, 1e5], sig=[100/2.35, 350/2.35])
        split = True
    else:
        f1_ = f1
        
    if 'fitResults_sigma' in f1.keys():
        f2 = tabular.ResultsFilter(f1_, error_x=[0, 30], A=[5, 1e5], sig=[100 / 2.35, 350 / 2.35])
    else:
        f2 = tabular.ResultsFilter(f1_, error_x=[0, 30], A=[5, 1e5])

    if 'fitResults_z0' in f1_.keys():
        threeD = True

    if 'Events' in dir(f1.h5f.root):
        events = f1.h5f.root.Events[:]

        evKeyNames = set()
        for e in events:
            evKeyNames.add(e['EventName'])

        if b'ProtocolFocus' in evKeyNames:
            stack = True



    xmax = f2['x'].max()
    ymax = f2['y'].max()

    if xmax > ymax:
        step = xmax/thumbSize
    else:
        step = ymax/thumbSize

    im, edx, edy = histogram2d(f2['x'], f2['y'], [arange(0, xmax, step), arange(0, ymax, step)])

    f1.close()

    im = minimum(2*(255*im)/im.max(), 255).T


    im = concatenate((im[:,:,newaxis], im[:,:,newaxis], im[:,:,newaxis]), 2)

    if stack:
        im[-10:, -10:, 0] = 180

    if threeD:
        im[-10:, -10:, 1] = 180

    if split:
        im[-10:-5, :10, 1] = 210
        im[-5:, :10, 0] = 210

    return im.astype('uint8')

if __name__ == '__main__':
    import gnomevfs
    inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
    outputFile = sys.argv[2]
    thumbSize = int(sys.argv[3])

    im = generateThumbnail(inputFile, thumbSize)

    Image.fromarray(im).save(outputFile, 'PNG')


