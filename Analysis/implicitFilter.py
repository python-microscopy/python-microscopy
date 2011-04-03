import numpy as np
from scipy import linalg


def impfilt(fcn, x0, args, maxIters=200, initStepSize=.01, minStepSize=.0005, maxFevals = 200):
    nIters = 0
    nFeval = 0
    stepsize = initStepSize

    x0 = np.array(x0, 'f')
    nDim = len(x0)

    fval = fcn(x0, *args)
    #print fval
    nFeval += 1

    #xts = []

    while nIters < maxIters and nFeval < maxFevals and stepsize > minStepSize:
        nIters += 1

        changed = False

        #print stepsize

        for i in range(nDim):
            #print i
            dx = stepsize
            xCand = x0.copy()
            xCand[i] = x0[i] + dx

            fCand = fcn(xCand, *args)
            #print xCand, fCand
            nFeval += 1

            if not fCand < fval: #try the other direction
                dx = -dx
                xCand[i] = x0[i] + dx
                fCand = fcn(xCand, *args)
                nFeval += 1

            while fCand < fval and nFeval < maxFevals:
                #search along this line, with this step size
                changed = True
                dx *= 2

                x0[:] = xCand[:]
                fval = fCand

                #xts.append(x0.copy())

                xCand[i] = x0[i] + dx
                fCand = fcn(xCand, *args)
                #print xCand,fCand
                nFeval += 1

        if not changed:
            stepsize *=.5

    print 'Optimisation terminated:'
    print 'nIterations: %d' % nIters
    print 'nFevals: %d' % nFeval

    return x0 #, xts


def impfilt2(fcn, x0, args, maxIters=200, initStepSize=.01, minStepSize=.0005, maxFevals = 200):
    nIters = 0
    nFeval = 0
    stepsize = initStepSize

    x0 = np.array(x0, 'f')
    nDim = len(x0)

    fval = fcn(x0, *args)
    #print fval
    nFeval += 1

    xv = np.zeros(3)
    fv = np.zeros(3)
    ons = np.ones(3)

    #xts = []

    changed = True

    while nIters < maxIters and nFeval < maxFevals and stepsize > minStepSize and changed:
        nIters += 1

        changed = False

        maxChange = 0

        #print stepsize

        for i in range(nDim):
            #print i
            dx = stepsize
            xCand = x0.copy()
            xCand[i] = x0[i] + dx

            xv[0] = x0[i]
            fv[0] = fval
            
            xv[1] = xCand[i]

            fCand = fcn(xCand, *args)
            fv[1] = fCand
            #print xCand, fCand
            nFeval += 1

            dfdx = (fCand - fval)/dx

            if fCand < fval: #may as well already update
               x0[:] = xCand[:]
               fval = fCand
               changed = True
               
            #try to overshoot 
            dx =  -np.sign(dfdx)*2*dx

            #print i, xv[0], xv[1], dfdx, dx

            xCand[i] = x0[i] + dx
            fCand = fcn(xCand, *args)
            nFeval += 1


            while fCand < fval and nFeval < maxFevals:
                #search along this line, with this step size
                #print 's'
                changed = True
                dx *= 2

                xv[0] = x0[i]
                fv[0] = fval

                x0[:] = xCand[:]
                fval = fCand

                xv[1] = x0[i]
                fv[1] = fval

                #xts.append(x0.copy())

                xCand[i] = x0[i] + dx
                fCand = fcn(xCand, *args)
                #print xCand,fCand
                nFeval += 1

            #now fit a parabola
            xv[2] = xCand[i]
            fv[2] = fCand

            #print np.vstack([xv**2, xv, ons])
            A, B, C = linalg.solve(np.hstack([(xv**2)[:, None], xv[:, None], ons[:, None]]), fv)

            #should be minimum
            xn = -B/(2*A)

            #try and see if this is better
            xCand[i] = xn
            fCand = fcn(xCand, *args)
            nFeval += 1

            #print xv[0],fval,  xn, fCand

            if fCand < fval:
                #print 'Accepting quad est.'
                x0[:] = xCand[:]
                fval = fCand
                changed = True

            maxChange = max(maxChange, abs(x0[i] - xv[0]))
            #print maxChange




        #if not changed:
        stepsize *=.1

    print 'Optimisation terminated:'
    print 'nIterations: %d' % nIters
    print 'nFevals: %d' % nFeval

    return x0 #, xts










