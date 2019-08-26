#!/usr/bin/python

###############
# edges.py
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
from .edgeDB import *
import numpy
import threading
#from PYME.ParallelTasks.launchWorkers import cpuCount
import multiprocessing

dtype = [('numIncidentEdges',  'i4'),('endVertices',  '7i4'),('edgeLengths',  '7f4'), ('meanNeighbourDist', 'f4'), ('nextRecordIndex',  'i4')]


class EdgeDB:
    def __init__(self, T, shm=False, extraSpaceFactor=1.5, calcDistances=True):
        self.Nverts = len(T.x)
        if shm:
            from PYME.util.shmarray import shmarray
            self.edgeArray = shmarray.zeros(int(extraSpaceFactor*self.Nverts), dtype)
        else:
            self.edgeArray = numpy.zeros(int(extraSpaceFactor*self.Nverts), dtype)

        #record how many vertices there are
        self.edgeArray[-1]['numIncidentEdges'] = self.Nverts

        #say where we can start adding extra rows
        self.edgeArray[-1]['nextRecordIndex'] = self.Nverts + 1

        addEdges(self.edgeArray, T.edges)

        if calcDistances:
            self.calcDistances((T.x, T.y))

    def getVertexEdgeLengths(self, i):
        return getVertexEdgeLengths(self.edgeArray, i)

    def getVertexNeighbours(self, i):
        return getVertexNeighbours(self.edgeArray, i)

    def calcDistances(self, coords, threads=True):
        if threads:
            N = len(coords[0])

            taskSize = int(N / multiprocessing.cpu_count())  # note that this floors
            taskEdges = list(range(0, N, taskSize)) + [N]

            tasks = [(taskEdges[i], taskEdges[i+1]) for i in range(len(taskEdges)-1)]
            #print tasks

            threads = [threading.Thread(target = calcEdgeLengths, args=(self.edgeArray, coords,  t[0], t[1])) for t in tasks]
            
            for p in threads:
                #print p
                p.start()

            for p in threads:
                p.join()

        else:
            calcEdgeLengths(self.edgeArray, coords)

    def getNeighbourDists(self):
        return self.edgeArray['meanNeighbourDist'][:self.Nverts]

    def segment(self, lenThresh):
        return segment(self.edgeArray, lenThresh)


def objectIndices(segmentation, minSize=3):
    objects = [];
    inds = numpy.arange(len(segmentation)).astype('i')

    nPts = numpy.histogram(segmentation, numpy.arange(segmentation.max()))[0]

    for i in numpy.where(nPts > minSize)[0]:
        #objects.append(numpy.where(segmentation == i)[0])
        objects.append(inds[segmentation == i])
        

    return objects




