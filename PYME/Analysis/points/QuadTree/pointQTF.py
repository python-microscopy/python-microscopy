#!/usr/bin/python

##################
# pointQT.py
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

QT_MAXRECORDS = 10 #maximum number of records in a leaf
QT_MAXDEPTH = 100 #let leaves grow larger if a maximum depth is exceeded (to prevent infinite recursion)

import numpy as np

from math import floor, ceil

class qtNode:
    def __init__(self,x0,x1, y0, y1, depth=0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)
        
        self.depth = depth
        self.numRecords = 0
        

    def insert(self,obj, x, y):
        return self

    #def get(self,x0, x1,y0, y1):
    #    return []

    def getAll(self):
        return self.get(self.x0, self.x1, self.y0, self.y1)

    #def drawTree(self, depth=0):
    #    return ''

    #def getLeaves(self, maxDepth=-1):
    #    return []



class qtRec:
    def __init__(self,x,y, obj):
        self.x = x
        self.y = y
        self.obj = obj

    def __str__(self):
        return 'qtRecord at (%f, %f) -> %s' % (self.x, self.y, repr(self.obj))


class qtLeafNode(qtNode):
    def __init__(self,x0,x1, y0, y1, depth=0):
        qtNode.__init__(self,x0,x1, y0, y1, depth)
        #self.x0 = x0
        #self.y0 = y0
        #self.x1 = x1
        #self.y1 = y1
        #self.records = []
        
        self.records = np.zeros([QT_MAXRECORDS, 3])
    
    def insert(self, rec):
        if self.numRecords < QT_MAXRECORDS or self.depth >= QT_MAXDEPTH: #don't need to subdivide
            self.records[self.numRecords, :] = rec
            self.numRecords += 1
            return self
        else: #subdivide 
            newBranch = qtBranchNode(self.x0,self.x1, self.y0, self.y1, self.depth)
            #for r in self.records: #copy records
            #       newBranch.insert(r)
            newBranch.insertRecords(self.records[:self.numRecords, :])
            newBranch.insert(rec)
            return newBranch
            
    def insertRecords(self, recs):
        nNew = len(recs)
        if (self.numRecords + nNew) < QT_MAXRECORDS or self.depth >= QT_MAXDEPTH: #don't need to subdivide
            self.records[self.numRecords:(self.numRecords + nNew), :] = recs
            self.numRecords += nNew
            return self
        else: #subdivide 
            newBranch = qtBranchNode(self.x0,self.x1, self.y0, self.y1, self.depth)
            #for r in self.records: #copy records
            newBranch.insertRecords(self.records[:self.numRecords, :])
            newBranch.insertRecords(recs)
            return newBranch

    def get(self,x0,x1, y0, y1):
        #return [r for r in self.records if (r.x > x0) and (r.x < x1) and (r.y > y0) and (r.y < y1)]
        r = self.records[:self.numRecords, :]
        return r[(r[:,0] > x0)*(r[:,0] < x1)*(r[:,1] > y0)*(r[:,1] < y1), :]

    def drawTree(self, depth=0):
        st = u'' + repr(self) +'\n'
        for r in self.records:
            for i in range(depth):
                st += u'\u2502 '
            st += u'\u251C\u2500' + str(r) +'\n'

        for i in range(depth):
                st += u'\u2502 '
        st += '\n'
        return st

    def getLeaves(self, maxDepth=-1):
        return [self]


class qtBranchNode(qtNode):
    def __init__(self,x0,x1, y0, y1, depth=0):
        qtNode.__init__(self,x0,x1, y0, y1, depth)
        self.xc = (x0 + x1)/2
        self.yc = (y0 + y1)/2
        #self.x0 = x0
        #self.y0 = y0
        #self.x1 = x1
        #self.y1 = y1

        #generate child nodes
        self.NW = qtLeafNode(x0,self.xc, y0, self.yc, depth+1)
        self.NE = qtLeafNode(self.xc,x1, y0, self.yc, depth+1)
        self.SE = qtLeafNode(self.xc,x1, self.yc, y1, depth+1)
        self.SW = qtLeafNode(x0,self.xc, self.yc, y1, depth+1)

    def insert(self, rec):
        if rec[0] < self.xc:
            if rec[1] < self.yc:
                self.NW = self.NW.insert(rec)
            else:
                self.SW = self.SW.insert(rec)
        else:
            if rec[1] < self.yc:
                self.NE = self.NE.insert(rec)
            else:
                self.SE = self.SE.insert(rec)

        self.numRecords += 1
        return self
        
    def insertRecords(self, recs):
        W = recs[:,0] < self.xc
        N = recs[:,1] < self.yc
        self.NW.insertRecords(recs[W*N, :])
        self.NE.insertRecords(recs[(W<1)*N, :])
        self.SE.insertRecords(recs[(W<1)*(N < 1), :])
        self.SW.insertRecords(recs[W*(N < 1), :])
        
        self.numRecords += recs.shape[0]
        return self

    def get(self,x0,x1, y0, y1):
        ret = []
        xc = (self.x0 + self.x1)/2
        yc = (self.y0 + self.y1)/2

        if x0 < xc:
            if y0 < yc:
                ret += self.NW.get(x0, x1, y0, y1)
            if y1 > yc:
                ret += self.SW.get(x0, x1, y0, y1)

        if x1 > xc:
            if y0 < yc:
                ret += self.NE.get(x0, x1, y0, y1)
            if y1 > yc:
                ret += self.SE.get(x0, x1, y0, y1)
        
        return ret

    def drawTree(self, depth=0):
        st = repr(self) +'\n'
        for i in range(depth):
            st += u'\u2502 '
        st += u'\u251C\u2500NW ' + self.NW.drawTree(depth+1)
        for i in range(depth):
            st += u'\u2502 '
        st += u'\u251C\u2500NE ' + self.NE.drawTree(depth+1) 
        for i in range(depth):
            st += u'\u2502 '
        st += u'\u251C\u2500SE ' + self.SE.drawTree(depth+1) 
        for i in range(depth):
            st += u'\u2502 '
        st += u'\u251C\u2500SW ' + self.SW.drawTree(depth+1)

        for i in range(depth):
                st += u'\u2502 '
        st += '\n'

        return st

    def getLeaves(self, maxDepth=-1):
        if maxDepth == 0:
            return [self]
        else:
            return self.NW.getLeaves(maxDepth -1) +self.NE.getLeaves(maxDepth -1) + self.SW.getLeaves(maxDepth -1) + self.SE.getLeaves(maxDepth -1) 
        

#root class - wraps root node changes when subdividing
class qtRoot(qtNode):
    def __init__(self,x0,x1, y0, y1):
        qtNode.__init__(self,x0,x1, y0, y1)
        self.root = qtLeafNode(x0,x1,y0,y1)

    def insert(self,rec):
        self.root = self.root.insert(rec)
        self.numRecords += 1

    #def get(self,x0, x1,y0, y1):
    #    return self.root.get(x0, x1,y0, y1)

    #def getLeaves(self, maxDepth=-1):
    #    return self.root.getLeaves(maxDepth)

    def __getattr__(self, name):
        return getattr(self.root, name)



def getInRadius(qt, x, y, radius):
    return [r for r in qt.get(x - radius, x + radius, y - radius, y + radius) if ((r.x - x)**2 + (r.y - y)**2) < radius**2]


def createQT(x,y, t=None):
    """ creates a quad tree from a list of x,y positions"""

    di = max(x.max() - x.min(), y.max() - y.min())
    qt = qtRoot(100*floor(x.min()/100), 100*ceil((x.min()+di)/100),100*floor(y.min()/100), 100*ceil((y.min()+di)/100))

    if t is None:
        for xi, yi in zip(x,y):
            qt.insert(np.array((xi,yi, 0)))
    else:
        for xi, yi, ti in zip(x,y, t):
            qt.insert(np.array((xi,yi, ti)))

    return qt
