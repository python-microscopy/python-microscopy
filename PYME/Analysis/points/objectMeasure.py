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

import numpy as np
# from matplotlib import tri
from scipy.spatial import Delaunay
from PYME.Analysis.points import gen3DTriangs
from PYME.Analysis.points import moments
from PYME.IO.MetaDataHandler import get_camera_roi_origin

def getPrincipalAxis(obj_c, numIters=10):
    """PCA via e.m. (ala wikipedia)"""
    X = obj_c.T
    p = np.random.rand(2)
    
    for i in range(numIters):
        t = (np.dot(p, X) * X).sum(1)
        p = t / np.linalg.norm(t)

    return p

def measureAligned(object, measurements = {}):
    obj_c = (object - np.mean(object, 0))

    A = np.matrix(obj_c[:,0]).T
    b = np.matrix(obj_c[:,1]).T

    #majorAxisGradient = float((linalg.inv(A.T*A)*A.T)*b)

    #measurements['majorAxisAngle'] = arctan(majorAxisGradient)

    #majorAxis = array([1, majorAxisGradient])
    #majorAxis = majorAxis/linalg.norm(majorAxis)

    majorAxis = getPrincipalAxis(obj_c)

    measurements['majorAxisAngle'] = np.arccos(majorAxis[0])

    minorAxis = majorAxis[::-1]*np.array([-1, 1])

    nx = (obj_c*majorAxis).sum(1)
    ny = (obj_c*minorAxis).sum(1)

    measurements['stdMajor'] = np.std(nx)
    measurements['stdMinor'] = np.std(ny)

    measurements['lengthMajor'] = nx.max() - nx.min()
    measurements['lengthMinor'] = ny.max() - ny.min()

    return measurements

measureDType = [('objID', 'i4'),('xPos', 'f4'), ('yPos', 'f4'), ('NEvents', 'i4'), ('Area', 'f4'),
    ('Perimeter', 'f4'), ('majorAxisAngle', 'f4'), ('stdMajor', 'f4'), ('stdMinor', 'f4'),
    ('lengthMajor', 'f4'), ('lengthMinor', 'f4'), ('moments', '25f4'), ('momentErrors', '25f4')]

def measure(object, min_edge_length, output=np.zeros(1, dtype=measureDType)):
    """
    Calculates a number of measurements for a collection of points (as outlined in dtype above).
    
    **Note:** All parameters except `Area` and `Perimeter` do not require any form of segmentation and are parameter
    free. `Area` and `Perimeter` are calculated using a Delaunay triangulation of the points. To allow accurate
    measurements of non-convex objects, triangles are cropped from the tessellation if their edge length
    exceeds a threshold given by `min_edge_length`. The value of `min_edge_length` determines a length scale over
    which concave regions of the object border are smoothed out. For large `min_edge_length` you get the
    convex hull of the object points. A reasonable value for min_edge_length is probably in the range
    10 <= min_edge_length <= 100, depending on object size, and localisation density.
    
    For objects with a small number of points, the `Area` and `Perimeter` values will be highly influenced by
    localisation stochasticity and unreliable. A rough rule of thumb is that you should have at least 50
    localisation events in an object before you can start to trust areas and perimeters. For lower event counts,
    the only (semi-)reliable metrics for object size are the std. deviations of the point positions
    (`stdMajor` and `stdMinor`) or the radius of gyration (TODO- ADD radius of gyration as a computed parameter).
    
    For roughly circular clusters, an approximate 'Area' parameter can be derived from these as follows:
     
     r = (2.35*((stdMajor + stdMinor)/2))/2 #calculate a FWHM based on the average std. dev, and divide by 2 to get a radius
     A = pi*r^2
     
    For elongated clusters, a rectangular model and the product or the major and minor axis FWHMs might be better.
    
    
    Parameters
    ----------
    object : shape [N, 2] numpy array
             x, y positions of object localisations
    min_edge_length: float
             edge length in nm at which to cull triangles from the point convex hull when estimating cluster areas
    output : [optional] pre-allocated output array

    Returns
    -------
    
    an ndarray with dtype=measureDType containing the object measurements. Note that entries which are impossible to
    compute (e.g. areas with nEvents < 3 and aligned measures with nEvents < 2) will be returned as 0.

    """
    #measurements = {}

    output['NEvents'] = object.shape[0]
    output['xPos'] = object[:, 0].mean()
    output['yPos'] = object[:, 1].mean()

    if object.shape[0] > 3:
        # T = tri.Triangulation(object.ravel(),2)
        T = Delaunay(object)
        P, A, triI = gen3DTriangs.cull_triangles_2D(T, min_edge_length)

        if not len(P) == 0:
            output['Area'] = A.sum() / 3

            #print triI

            extEdges = gen3DTriangs.getExternalEdges(triI)

            output['Perimeter'] = gen3DTriangs.getPerimeter(extEdges, T)
        else:
            output['Area'] = 0
            output['Perimeter'] = 0

        ms, sm = moments.calcMCCenteredMoments(object[:,0], object[:,1])
        #print ms.ravel()[3]
        #print ms.shape, measurements['moments'].shape
        output['moments'][:] = ms.ravel()
        output['momentErrors'][:] = sm.ravel()

    if object.shape[0] > 1:
        measureAligned(object, output)
    #measurements.update(measureAligned(object))
    return output


def measureObjects(objects, min_edge_length):
    measurements = np.zeros(len(objects), dtype=measureDType)

    for i, obj in enumerate(objects):
        measure(obj, min_edge_length, measurements[i])

    return measurements

def measureObjectsByID(filter, min_edge_length, ids, key='objectID'):
    x = filter['x'] #+ 0.1*random.randn(filter['x'].size)
    y = filter['y'] #+ 0.1*random.randn(x.size)
    # id = filter['objectID'].astype('i')
    id = filter[key].astype('i')

    #ids = set(ids)

    measurements = np.zeros(len(ids), dtype=measureDType)

    for j,i in enumerate(ids):
        if not i == 0:
            ind = id == i
            obj = np.vstack([x[ind],y[ind]]).T
            #print obj.shape
            measure(obj, min_edge_length, measurements[j])
            measurements[j]['objID'] = i

    return measurements

def calcEdgeDists(objects, objMeasures):
    T = tri.Triangulation(np.array([objMeasures['xPos'], objMeasures['yPos']]).T,2)

    va = np.array(T.set)
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

            d = np.sqrt(dx**2 + dy**2)

            ed = np.min(ed, d.min())

        minEdgeDists.append(ed)

    return np.array(minEdgeDists)
