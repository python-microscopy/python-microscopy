#!/usr/bin/python

###############
# segment.py
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
from . import edges
import numpy

def collectConnected(edb, vertexNum, visited, lenThresh, objectNum):
    connected = []

    neighbours = edb.getVertexNeighbours(vertexNum)
    neighbourDists = edb.getVertexEdgeLengths(vertexNum)

    for conn in neighbours[neighbourDists < lenThresh]:
        if not visited[conn]:
            visited[conn] = objectNum
               
            #connected.append(conn)
            #connected +=
            collectConnected(edb, conn, visited, lenThresh, objectNum)

    #return connected


def segment(edb, lenThresh, minSize=3):
    objects = []
    
    visited = numpy.zeros(edb.Nverts, 'int32')

#    if minSize == None:
#        minSize = va.shape[1] + 1 #only return objects which have enough points to be a volume

    vertexNum = 0
    objectNum = 1
    #objectIndices = numpy.zeros(edb.Nverts, 'int32')

    while (visited == 0).sum(): #while there are still vertices we haven't visited
        while visited[vertexNum]  > 0 and vertexNum < visited.shape[0]: #skip over previously visited vertices
            vertexNum += 1

        #flag current vertex as visited
        visited[vertexNum] = objectNum
        #print vertexNum, objectNum
        # and add to object
        #obj = [vertexNum]

        #con =
        collectConnected(edb, vertexNum, visited, lenThresh, objectNum)

        objectNum += 1

#        obj += con
#
#        if len(obj) > minSize:
#            objects.append(numpy.array(obj))

    return visited, objectNum