#!/usr/bin/python

##################
# QTrend.py
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

import matplotlib
#from pylab import *
import numpy as np
from matplotlib import pyplot as plt


def rendQT(ax, qt, maxDepth=-1, cmap=plt.cm.hot, maxIsc = 1):
    ax.cla()
    ax.set_xlim(qt.x0,qt.x1)
    ax.set_ylim(qt.y0, qt.y1)

    lvs = qt.getLeaves(maxDepth)

    Is = [float(l.numRecords)*2**(2*l.depth) for l in lvs]

    maxI = max(Is)
    print(maxI)

    for l in lvs:
        c = cmap((float(l.numRecords)*2**(2*l.depth))/(maxI*maxIsc))
        ax.add_patch(matplotlib.patches.Rectangle((l.x0,l.y0), l.x1- l.x0, l.y1 - l.y0, fc= c, ec = c))

    plt.show()

def rendQTe(ax, qt, maxDepth=-1, cmap=plt.cm.hot, maxIsc = 1):
    ax.cla()
    ax.set_xlim(qt.x0,qt.x1)
    ax.set_ylim(qt.y0, qt.y1)

    lvs = qt.getLeaves(maxDepth)

    Is = [float(l.numRecords)*2**(2*l.depth) for l in lvs]

    maxI = max(Is)
    print(maxI)

    for l in lvs:
        c = cmap((float(l.numRecords)*2**(2*l.depth))/(maxI*maxIsc))
        ax.add_patch(matplotlib.patches.Rectangle((l.x0,l.y0), l.x1- l.x0, l.y1 - l.y0, fc='w', ec = 'k'))

    plt.show()

def rendQTa(ar, qt, maxDepth=100):
    (sX, sY) = ar.shape    
    
    maxDepth = min(maxDepth, np.floor(np.log2(min(sX, sY))))

    lvs = qt.getLeaves(maxDepth)

    dX = float(qt.x1 - qt.x0)/sX
    dY = float(qt.y1 - qt.y0)/sY

    #ensure aspect ratio is OK
    dX = max(dX, dY)
    dY = dX

    print(dX)

    for l in lvs:
        dx = l.x1 - l.x0
        #if dx < dX:
        #    print dx
        ar[int(round((l.x0 - qt.x0)/dX)):int(round((l.x1 - qt.x0)/dX)),
       int(round((l.y0 - qt.y0)/dY)):int(round((l.y1 - qt.y0)/dY))] = \
       (float(l.numRecords)*2**(2*(l.depth - maxDepth)))

    #return ar

# normalized version: number of events per pixel, ar.sum should be equal to total number of events
# only approximate due to rounding at pixel boundaries
def rendQTan(ar, qt, maxDepth=100):
    (sX, sY) = ar.shape    
    
    maxDepth = min(maxDepth, np.floor(np.log2(min(sX, sY))))

    lvs = qt.getLeaves(maxDepth)

    dX = float(qt.x1 - qt.x0)/sX
    dY = float(qt.y1 - qt.y0)/sY

    #ensure aspect ratio is OK
    dX = max(dX, dY)
    dY = dX

    print(dX)

    ar[:,:] = 0 # initialize
    numevts = 0
    nevts2 = 0
    for l in lvs:
        ix0 = round((l.x0 - qt.x0)/dX)
        ix1 = round((l.x1 - qt.x0)/dX)
        iy0 = round((l.y0 - qt.y0)/dY)
        iy1 = round((l.y1 - qt.y0)/dY)
        ar[ix0:ix1,iy0:iy1] += float(l.numRecords)/float((ix1-ix0)*(iy1-iy0))


class QTRendererNode:
    def __init__(self, node, i=0):
        self.children = []
        self.type = 'branch'
        self.width=0.0
        self.occupancy = node.numRecords
        self.i = i #record order in which nodes are visited (for colouring)

        self.node = node
        
        if 'records' in dir(node): #is a leaf
            self.type = 'leaf'
            #self.occupancy = node.numRecords
            #self.width += self.occupancy
            self.width = 2.0
            self.i = i +1
        else: #is a branch
            #nodeWidth = 0
            for n in [node.NW, node.NE, node.SW, node.SE]:
                nr = QTRendererNode(n, self.i)
                self.width += nr.width
                #nodeWidth = max(nodeWidth, nr.width)
                self.children.append(nr)
                self.i = nr.i
            
            #self.width = 4*nodeWidth
            self.width +=5

    def drawNode(self, ax, offset=(0,0), iMax=None):
        x0,y0 = offset

        if iMax is None:
            iMax = self.i

        if self.type == 'leaf':
            ax.plot([x0, x0], [y0,y0-5], c=plt.cm.hsv(float(self.i)/iMax), lw=(np.log2(self.occupancy+1.0)/2 +1.0))

            y0-= 5

            #ax.text(x0, y0 + 1, '%d' % self.occupancy)
        else:
            ax.plot([x0, x0], [y0,y0-10], 'k',lw=(np.log2(self.occupancy+1.0)/2 +1.0))

            y0-= 10

            xNW = x0 - (self.width/2 - self.children[0].width/2) + 1
            xNE = xNW + self.children[0].width/2 + self.children[1].width/2 + 1
            xSW = xNE + self.children[1].width/2 + self.children[2].width/2 + 1
            xSE = x0 + (self.width/2 - self.children[3].width/2) - 1

            #print self.width, xSE - x0
            
            
            #xNW = x0 - (self.width/2 - self.width/8)-2
            #xNE = xNW + self.width/4 + 1
            #xSW = xNE + self.width/4 + 1
            #xSE = x0 + (self.width/2 - self.width/8)+2

            #ax.plot([xNW, xSE], [y0,y0], 'k',lw=(log2(self.occupancy+1.0)+1.0)/2.0 )
            
            for ch, xp in zip(self.children, [xNW, xNE, xSW, xSE]):
                ch.drawNode(ax, (xp, y0), iMax)

            ax.plot([xNW, xSE], [y0,y0], 'k',lw=(np.log2(self.occupancy+1.0)/2+1.0) )


    def drawNodeDL(self, ax1, ax2, offset=(0,0), iMax=None):
        x0,y0 = offset

        if iMax is None:
            iMax = self.i

        if self.type == 'leaf':
            ax1.plot([x0, x0], [y0,y0-5], c=plt.cm.hsv(float(self.i)/iMax), lw=(np.log2(self.occupancy+1.0)/2 +1.0))

            y0-= 5

            ax2.add_patch(matplotlib.patches.Rectangle((self.node.x0,self.node.y0), self.node.x1- self.node.x0, self.node.y1 - self.node.y0, fc='w', ec = plt.cm.hsv(float(self.i)/iMax) ))

            #ax.text(x0, y0 + 1, '%d' % self.occupancy)
        else:
            ax1.plot([x0, x0], [y0,y0-10], 'k',lw=(np.log2(self.occupancy+1.0)/2 +1.0))

            y0-= 10

            xNW = x0 - (self.width/2 - self.children[0].width/2) + 1
            xNE = xNW + self.children[0].width/2 + self.children[1].width/2 + 1
            xSW = xNE + self.children[1].width/2 + self.children[2].width/2 + 1
            xSE = x0 + (self.width/2 - self.children[3].width/2) - 1

            #print self.width, xSE - x0
            
            
            #xNW = x0 - (self.width/2 - self.width/8)-2
            #xNE = xNW + self.width/4 + 1
            #xSW = xNE + self.width/4 + 1
            #xSE = x0 + (self.width/2 - self.width/8)+2

            #ax.plot([xNW, xSE], [y0,y0], 'k',lw=(log2(self.occupancy+1.0)+1.0)/2.0 )
            
            for ch, xp in zip(self.children, [xNW, xNE, xSW, xSE]):
                ch.drawNodeDL(ax1,ax2, (xp, y0), iMax)

            ax1.plot([xNW, xSE], [y0,y0], 'k',lw=(np.log2(self.occupancy+1.0)/2+1.0) )
            
            
        
            
            
        
    
    
