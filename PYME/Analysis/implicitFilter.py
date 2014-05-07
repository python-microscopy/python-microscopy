#!/usr/bin/python

###############
# implicitFilter.py
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

    print('Optimisation terminated:')
    print(('nIterations: %d' % nIters))
    print(('nFevals: %d' % nFeval))

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

    print('Optimisation terminated:')
    print(('nIterations: %d' % nIters))
    print(('nFevals: %d' % nFeval))

    return x0 #, xts


def impfilt3(fcn, x0, args, maxIters=200, initStepSize=.01, minStepSize=.0005, maxFevals = 200):
    nIters = 0
    nFeval = 0
    stepsize = initStepSize

    x0 = np.array(x0, 'f')
    nDim = len(x0)

    fval = fcn(x0, *args)
    #print fval
    nFeval += 1

    tv = np.zeros(3)
    fv = np.zeros(3)
    ons = np.ones(3)

    dfdx = np.zeros(2)

    #xts = []

    changed = True

    while nIters < maxIters and nFeval < maxFevals and stepsize > minStepSize and changed:
        nIters += 1

        changed = False

        maxChange = 0

        #print stepsize

        #find gradient
        for i in range(nDim):
            dx = stepsize
            xCand = x0.copy()
            xCand[i] = x0[i] + dx

            fCand = fcn(xCand, *args)
            nFeval += 1

            dfdx[i] = (fCand - fval)/dx

            if fCand < fval: #may as well already update
               x0[:] = xCand[:]
               fval = fCand
               changed = True

        dfdx_hat = dfdx/linalg.norm(dfdx)

        x_0 = x0.copy()
        B = -linalg.norm(dfdx)
        C = fval
        #t = 0

        #print linalg.norm(dfdx), linalg.norm(dfdx_hat)
        #print dfdx_hat, dfdx

        #try to overshoot
        dxv =  -dfdx_hat*2*dx
        t = 2*dx

        #print i, xv[0], xv[1], dfdx, dx

        fPred = fval + t*B
        
        xCand = x0 + dxv
        fCand = fcn(xCand, *args)
        nFeval += 1

        print((x0, xCand, fPred, fCand))


        while fCand < fval and nFeval < maxFevals:
            #search along this line, with this step size
            #print 's'
            changed = True

            #print B
            B = (fval-fCand)/t
            #print B

            x0[:] = xCand[:]
            fval = fCand

            C = fval
            
            dxv *= 2
            t = 2*dx

            #xts.append(x0.copy())

            xCand = x0 + dxv
            fCand = fcn(xCand, *args)
            fPred = fval + t*B
            #print xCand,fCand
            nFeval += 1

#        #now fit a parabola
#        xv[2] = xCand[i]
#        fv[2] = fCand
#
#        #print np.vstack([xv**2, xv, ons])
#        A, B, C = linalg.solve(np.hstack([(xv**2)[:, None], xv[:, None], ons[:, None]]), fv)

        #print t
        A = (-C - B*t)/t**2

        print((A, B, C))

        #should be minimum
        tn = -B/(2*A)

        #try and see if this is better
        xCand = x0 + tn*dfdx_hat
        fCand = fcn(xCand, *args)
        nFeval += 1

        print(('q\t', tn,  x0, xCand, fval, fCand))

        if fCand < fval:
            print('Accepting quad est.')
            x0[:] = xCand[:]
            fval = fCand
            changed = True

        #maxChange = max(maxChange, abs(x0[i] - xv[0]))
        #print maxChange




        #if not changed:
        stepsize *=.1

    print('Optimisation terminated:')
    print(('nIterations: %d' % nIters))
    print(('nFevals: %d' % nFeval))

    return x0 #, xts









