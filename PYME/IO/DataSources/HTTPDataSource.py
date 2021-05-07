#!/usr/bin/python

##################
# HDFDataSource.py
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

#from PYME.ParallelTasks.relativeFiles import getFullFilename
#import tables
from .BaseDataSource import XYTCDataSource

#import httplib
#import urllib
import requests
try:
    # noinspection PyCompatibility
    import cPickle as pickle
except:
    #py3
    import pickle
    
import time

SHAPE_LIFESPAN = 5



class DataSource(XYTCDataSource):
    moduleName = 'HTTPDataSource'
    def __init__(self, url, queue=None):
        self.url = url
        self.lastShapeTime = 0
        
        self.dshape = self._request('%s/SHAPE' % self.url)
        self.lastShapeTime = time.time()
    
    def _request(self, url):
        #f = urllib.urlopen(url)
        r = requests.get(url)
        data= pickle.loads(r.content)
        #f.close()
        return data

    def getSlice(self, ind):
        sliceURL = '%s/DATA/%d' % (self.url, ind)
        return self._request(sliceURL).squeeze()

    def getSliceShape(self):
        return tuple(self.dshape[:2])
        
    def getNumSlices(self):
        t = time.time()
        if (t-self.lastShapeTime) > SHAPE_LIFESPAN:
            self.dshape = self._request('%s/SHAPE' % self.url)
            self.lastShapeTime = t
            
        return self.dshape[2]

    def getEvents(self):
        return self._request('%s/EVENTS'%self.url)
        
    def getMetadata(self):
        return self._request('%s/METADATA'%self.url)
 
