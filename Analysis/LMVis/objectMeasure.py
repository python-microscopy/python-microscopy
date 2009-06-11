from numpy import *
import delaunay
import gen3DTriangs

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

measureDType = [('xPos', 'f4'), ('yPos', 'f4'), ('NEvents', 'i4'), ('Area', 'f4'),
    ('Perimeter', 'f4'), ('majorAxisAngle', 'f4'), ('stdMajor', 'f4'), ('stdMinor', 'f4'),
    ('lengthMajor', 'f4'), ('lengthMinor', 'f4')]

def measure(object, sizeCutoff, measurements = zeros(1, dtype=measureDType)):
    #measurements = {}

    measurements['NEvents'] = object.shape[0]
    measurements['xPos'] = object[:,0].mean()
    measurements['yPos'] = object[:,1].mean()

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

    measureAligned(object, measurements)
    #measurements.update(measureAligned(object))
    return measurements


def measureObjects(objects, sizeCutoff):
    measurements = zeros(len(objects), dtype=measureDType)

    for i, obj in enumerate(objects):
        measure(obj, sizeCutoff, measurements[i])

    return measurements

