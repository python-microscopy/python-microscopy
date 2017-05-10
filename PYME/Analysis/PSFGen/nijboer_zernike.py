#!/usr/bin/python

###############
# nijboer_zernike.py
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
from scipy.misc import comb
from scipy.special import jn
from six.moves import xrange

def vlj(l,j,m_a,p, q):
    
    vlj = (-1**p)*(m_a + l + 2*j)
    vlj *= comb(m_a + j + l - 1, l-1)
    vlj *= comb(j+l-1, l-1)
    vlj *= comb(l-1, p - j)
    vlj /= comb(q+l+j, l)
    
    return vlj


def V(m,n, r, f, lmax):
    #em = -1 for m odd and negative, 1 otherwise 
    if m < 0 and (m % 2):
        em = -1
    else:
        em = 1
        
    V = 0*r
    #l = arange(1, lmax)
    
    mu = 2*np.pi*r
    
    m_a = abs(m)
    p = (n - m_a)/2
    q = (n + m_a)/2
    
    s1 = em*np.exp(1j*f)
    
    for l in xrange(1, lmax):
        t1 = 0*r
        for j in xrange(p + 1):
            t1 += vlj(l,j,m_a,p, q)*jn(m_a + l + 2*j, mu)/(l*mu**l)
            
        V+= ((-2j*f)**(l-1))*t1
        
    return s1*V
    
def U(r, phi, f, Bcoeffs, lmax=10):
        U = 0*r
        
        for B, m, n in Bcoeffs:
            m_a = abs(m)
            U += B*(1j**m_a)*V(m_a,n, r, f, lmax)*np.exp(1j*m*phi)
            
        return 2*U
    
    
    