#!/usr/bin/python

###############
# kdFit.py
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


from PYME.Analysis._fithelpers import *
#from pylab import *
import matplotlib.pyplot as plt
import scipy.special


def efunc(p, c):
    Rbound, Runbound, c0,sig = p 
    return (Rbound - Runbound)*(1+ scipy.special.erf((np.log10(c) - c0)/sig))/2 + Runbound
    

def fitKD(filename):
    d = np.loadtxt(filename)
    #print d.shape
    if d.shape[0] == 2:
        c, f = d
    else:
        c,f = d.T
    
    plt.semilogx(c, f, 'x')
    
    r = FitModel(efunc, [10, 1, 0, 1], f, c)
    print(("""
    Fit Results:
    --------------------
     R_bound   = %3.2f
     R_unbound = %3.2f
     C50       = %3.3f
     sigma     = %3.2f
     
     """ % (r[0][0], r[0][1], 10**r[0][2], r[0][3])))
        
    
    #print log10(c[0] + .01), log10(c[-1])
    #print f[0]
    c2 = np.logspace(np.log10(c[0] + .01), np.log10(c[-1]), 100)
    plt.semilogx(c2, efunc(r[0],c2))
    
    plt.xlabel('Concentration')
    plt.ylabel('Ratio')
    
    plt.figtext(.6,.2, 'C50 = %3.3f' % 10**r[0][2])
    plt.show()

if __name__ == '__main__':
    import sys
    plt.ioff()
    fitKD(sys.argv[1])