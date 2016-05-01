#!/usr/bin/python

##################
# objectMeasure.py
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

from numpy import *
import delaunay
import gen3DTriangs
from PYME.Analysis import moments

def getPrincipalAxis(obj_c, numIters=10):
    '''PCA via e.m. (ala wikipedia)'''
    X = obj_c.T
    p = random.rand(2)
    
    for i in range(numIters):
        t = (dot(p, X) * X).sum(1)
        p = t / linalg.norm(t)

    return p

def measureAligned(object, measurements = {}):
    obj_c = (object - mean(object, 0))

    A = matrix(obj_c[:,0]).T
    b = matrix(obj_c[:,1]).T

    #majorAxisGradient = float((linalg.inv(A.T*A)*A.T)*b)

    #measurements['majorAxisAngle'] = arctan(majorAxisGradient)

    #majorAxis = array([1, majorAxisGradient])
    #majorAxis = majorAxis/linalg.norm(majorAxis)

    majorAxis = getPrincipalAxis(obj_c)

    measurements['majorAxisAngle'] = arccos(majorAxis[0])

    minorAxis = majorAxis[::-1]*array([-1, 1])

    nx = (obj_c*majorAxis).sum(1)
    ny = (obj_c*minorAxis).sum(1)

    measurements['stdMajor'] = std(nx)
    measurements['stdMinor'] = std(ny)

    measurements['lengthMajor'] = nx.max() - nx.min()
    measurements['lengthMinor'] = ny.max() - ny.min()

    return measurements

measureDType = [('objID', 'i4'),('xPos', 'f4'), ('yPos', 'f4'), ('NEvents', 'i4'), ('Area', 'f4'),
    ('Perimeter', 'f4'), ('majorAxisAngle', 'f4'), ('stdMajor', 'f4'), ('stdMinor', 'f4'),
    ('lengthMajor', 'f4'), ('lengthMinor', 'f4'), ('moments', '25f4'), ('momentErrors', '25f4')]

def measure(object, sizeCutoff, measurements = zeros(1, dtype=measureDType)):
    #measurements = {}

    measurements['NEvents'] = object.shape[0]
    measurements['xPos'] = object[:,0].mean()
    measurements['yPos'] = object[:,1].mean()

    if object.shape[0] > 3:
        T = delaunay.Triangulation(object.ravel(),2)
        P, A, triI = gen3DTriangs.gen2DTriangsTF(T, sizeCutoff)

        if not len(P) == 0:
            measurements['Area'] = A.sum()/3

            #print triI

            extEdges = gen3DTriangs.getExternalEdges(triI)

            measurements['Perimeter'] = gen3DTriangs.getPerimeter(extEdges, T)
        else:
            measurements['Area'] = 0
            measurements['Perimeter'] = 0

        ms, sm = moments.calcMCCenteredMoments(object[:,0], object[:,1])
        #print ms.ravel()[3]
        #print ms.shape, measurements['moments'].shape
        measurements['moments'][:] = ms.ravel()
        measurements['momentErrors'][:] = sm.ravel()

    if object.shape[0] > 1:
        measureAligned(object, measurements)
    #measurements.update(measureAligned(object))
    return measurements


def measureObjects(objects, sizeCutoff):
    measurements = zeros(len(objects), dtype=measureDType)

    for i, obj in enumerate(objects):
        measure(obj, sizeCutoff, measurements[i])

    return measurements

def measureObjectsByID(filter, sizeCutoff, ids):
    x = filter['x'] #+ 0.1*random.randn(filter['x'].size)
    y = filter['y'] #+ 0.1*random.randn(x.size)
    id = filter['objectID'].astype('i')

    #ids = set(ids)

    measurements = zeros(len(ids), dtype=measureDType)

    for j,i in enumerate(ids):
        if not i == 0:
            ind = id == i
            obj = vstack([x[ind],y[ind]]).T
            #print obj.shape
            measure(obj, sizeCutoff, measurements[j])
            measurements[j]['objID'] = i

    return measurements

def calcEdgeDists(objects, objMeasures):
    T = delaunay.Triangulation(array([objMeasures['xPos'], objMeasures['yPos']]).T,2)

    va = array(T.set)
    objInd = {}

    #dictionary mapping vertices to indicex
    for i in range(len(T.set)):
        #print tuple(T.set[i])
        objInd[tuple(va[i, :])] = i

    minEdgeDists = []

    for o, m in zip(objects, va):
        ed = 1e50
        for N in T.neighbours[tuple(m)]:
            iN = objInd[N]
            dx = o[:,0][:,None] - objects[iN][:,0][None,:]
            dy = o[:,1][:,None] - objects[iN][:,1][None,:]

            d = sqrt(dx**2 + dy**2)

            ed = min(ed, d.min())

        minEdgeDists.append(ed)

    return array(minEdgeDists)

