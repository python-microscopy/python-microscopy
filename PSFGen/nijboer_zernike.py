# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:22:17 2012

@author: david
"""

import numpy as np
from scipy.misc import comb
from scipy.special import jn

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
    
    
    