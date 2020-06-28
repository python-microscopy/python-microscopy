#!/usr/bin/python

##################
# ofind_nofilt.py
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

import scipy
import scipy.ndimage as ndimage
# import pylab

class OfindPoint:
    def __init__(self, x, y, z=None, detectionThreshold=None):
        """Creates a point object, potentially with an undefined z-value."""
        self.x = x
        self.y = y
        self.z = z
        self.detectionThreshold = detectionThreshold

#a hack so we'll be able to do something like ObjectIdentifier.x[i]
class PseudoPointList:
    def __init__(self,parent, varName):
        self.parent = parent
        self.varName = varName

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, key):
        tmp = self.parent[key]
        if not '__len__' in dir(tmp):
            return tmp.__dict__[self.varName]
        else:
            tm2 = []
            for it in tmp:
                tm2.append(it.__dict__[self.varName])
            return tm2

    #def __iter__(self):
    #    self.curpos = -1
    #    return self

    #def next(self):
    #    curpos += 1
    #    if (curpos >= len(self)):
    #        raise StopIteration
    #    return self[self.curpos]

class ObjectIdentifier(list):
    def __init__(self, data):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data)"""

        self.data = data

    
    def __ProjData(self):
        #project data
        if len(self.data.shape) == 2: #if already 2D, do nothing
            projData = self.data
        else:
            projData = self.data.max(2) 

        return projData


    def FindObjects(self, threshold, mask=None):
        """Finds objects using a simple threshold"""
        
        #save a copy of the parameters.
        self.threshold = threshold
        

        #clear the list of previously found points
        del self[:]
        
        #do filtering
        projData = self.__ProjData()
        
        #apply mask
        if not (self.mask is None):
            maskedData = projData*self.mask
        else:
            maskedData = projData

        #manually mask the edge pixels
        maskedData[0:5, 0:5] = 0
        maskedData[0:5, -5:] = 0
        maskedData[-5:, -5:] = 0
        maskedData[-5:, 0:5] = 0
        
        
        X,Y = scipy.mgrid[0:maskedData.shape[0], 0:maskedData.shape[1]]
    
        (labeledPoints, nLabeled) = ndimage.label(maskedData > self.threshold)

        objSlices = ndimage.find_objects(labeledPoints)

        #loop over objects
        for i in range(nLabeled):
            #measure position
            #x,y = ndimage.center_of_mass(im, labeledPoints, i)
            imO = im[objSlices[i]]
            x = (X[objSlices[i]]*imO).sum()/imO.sum()
            y = (Y[objSlices[i]]*imO).sum()/imO.sum()
            #and add to list
            self.append(OfindPoint(x,y,detectionThreshold=self.lowerThreshold))
        

        #create pseudo lists to allow indexing along the lines of self.x[i]
        self.x = PseudoPointList(self, 'x')
        self.y = PseudoPointList(self, 'y')
