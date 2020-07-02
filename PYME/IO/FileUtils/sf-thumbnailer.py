#!/usr/bin/python

##################
# sf-thumbnailer.py
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
#LOG_FILENAME = '/tmp/sf-thumbnailer.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)

import sys
import gnomevfs

try:
    # noinspection PyCompatibility
    import cPickle
except ImportError:
    #py3
    import pickle as cPickle

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    
    # from pylab import *
    from matplotlib.pyplot import *
    from numpy import *
    
    dpi = 100.
    
    
    inputFile = gnomevfs.get_local_path_from_uri(sys.argv[1])
    outputFile = sys.argv[2]
    thumbSize = int(sys.argv[3])
    
    #logging.debug('Input File: %s\n' % inputFile)
    #logging.debug('Ouput File: %s\n' % outputFile)
    #logging.debug('Thumb Size: %s\n' % thumbSize)
    
    #def generateThumbnail(inputFile, thumbsize):
    fid = open(inputFile)
    spx, spy = cPickle.load(fid)
    fid.close()
    
    f = figure(figsize=(thumbSize/dpi, 0.5*thumbSize/dpi))
    
    axes([0, 0, 1, 1])
    xin, yin = meshgrid(arange(0, 512*70, 4000), arange(0, 256*70, 4000))
    xin = xin.ravel()
    yin = yin.ravel()
    quiver(xin, yin, spx.ev(xin, yin), spy.ev(xin, yin), scale=2e3)
    xticks([])
    yticks([])
    axis('image')
    
    f.savefig(outputFile, dpi=dpi, format='png')

