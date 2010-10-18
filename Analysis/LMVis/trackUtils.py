import numpy

def calcTrackVelocity(x, y, ci):
    #if not self.init:
    #    self.InitGL()
    #    self.init = 1

    I = ci.argsort()

    v = numpy.zeros(x.shape) #velocities
    w = numpy.zeros(x.shape) #weights

    x = x[I]
    y = y[I]
    ci = ci[I]

    dists = sqrt(numpy.diff(x)**2 + numpy.diff(y)**2)

    #now calculate a mask so that we only include distances from within the trace
    mask = numpy.diff(ci) < 1

    #a points velocity is the average of the steps in each direction
    v[1:] += dists*mask
    v[:-1] += dists*mask

    #calulate weights to divide velocities by (end points only have one contributing step)
    w[1:] += mask
    w[:-1] += mask

    #leave velocities for length 1 traces as zero (rather than 0/0
    v[w > 0] = v[w > 0]/w[w > 0]

    #reorder
    v[I] = v

    return v





