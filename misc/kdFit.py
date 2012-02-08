# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:51:05 2012

@author: david
"""

from PYME.Analysis._fithelpers import *
from pylab import *
import scipy.special


def efunc(p, c):
    Rbound, Runbound, c0,sig = p 
    return (Rbound - Runbound)*(1+ scipy.special.erf((log10(c) - c0)/sig))/2 + Runbound
    

def fitKD(filename):
    d = np.loadtxt(filename)
    #print d.shape
    if d.shape[0] == 2:
        c, f = d
    else:
        c,f = d.T
    
    semilogx(c, f, 'x')
    
    r = FitModel(efunc, [10, 1, 0, 1], f, c)
    print '''
    Fit Results:
    --------------------
     R_bound   = %3.2f
     R_unbound = %3.2f
     C50       = %3.3f
     sigma     = %3.2f
     
     ''' % (r[0][0], r[0][1], 10**r[0][2], r[0][3])
        
    
    #print log10(c[0] + .01), log10(c[-1])
    #print f[0]
    c2 = logspace(log10(c[0] + .01), log10(c[-1]), 100)
    semilogx(c2, efunc(r[0],c2))
    
    xlabel('Concentration')
    ylabel('Ratio')
    
    figtext(.6,.2, 'C50 = %3.3f' % 10**r[0][2])
    show()

if __name__ == '__main__':
    import sys
    ioff()
    fitKD(sys.argv[1])