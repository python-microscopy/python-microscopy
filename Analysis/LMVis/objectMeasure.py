from numpy import *

def measureAligned(object):
    measurements = {}
    obj_c = (object - mean(object, 0))

    A = matrix(obj_c[:,0]).T
    b = matrix(obj_c[:,1]).T

    majorAxisGradient = float((inv(A.T*A)*A.T)*b)

    measurements['majorAxisAngle'] = atan(majorAxisGradient)

    majorAxis = array([1, majorAxisGradient])
    majorAxis = majorAxis/linalg.norm(majorAxis)

    minorAxis = majorAxis[::-1]*array([-1, 1])

    nx = (obj_c*majorAxis).sum(1)
    ny = (obj_c*minorAxis).sum(1)

    measurements['stdMajor'] = std(nx)
    measurements['stdMinor'] = std(ny)

    measurements['lengthMajor'] = nx.max() - nx.min()
    measurements['lengthMinor'] = ny.max() - ny.min()

    return measurements
