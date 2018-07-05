#from http://mri.brechmos.org/2010/01/19/anisotropic-diffusion-image-filtering-in-mri/
#from pylab import *
import numpy as np

def aniso(v, kappa=-1, N=1):

        if kappa == -1:
                kappa = np.prctile(v, 40)

        vf = v.copy()

        for ii in range(N):
                dE = -vf + np.roll(vf,-1,0)
                dW = vf - np.roll(vf,1,0)

                dN = -vf + np.roll(vf,-1,1)
                dS = vf - np.roll(vf,1,1)

                if len(v.shape) > 2:
                        dU = -vf + np.roll(vf,-1,2)
                        dD = vf - np.roll(vf,1,2)

                vf = vf + \
                        3./28. * ((np.exp(- (abs(dE) / kappa)**2 ) * dE) - (np.exp(- (abs(dW) / kappa)**2 ) * dW)) + \
                        3./28. * ((np.exp(- (abs(dN) / kappa)**2 ) * dN) - (np.exp(- (abs(dS) / kappa)**2 ) * dS))
                if len(v.shape) > 2:
                        vf += 1./28. * ((np.exp(- (abs(dU) / kappa)**2 ) * dU) - (np.exp(- (abs(dD) / kappa)**2 ) * dD))

        return vf