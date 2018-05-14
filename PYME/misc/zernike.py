#!/usr/bin/python

###############
# zernike.py
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
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:34:31 2011

@author: dbad004
"""
#import numpy as np
from numpy import cos, sin, isreal

zByName = {}
zByNumber = {}
NameByNumber = {}

def z(n, expr, descr):
    f = eval('lambda r, theta : ' + expr)
    zByName[descr] = f
    zByNumber[n] = f
    NameByNumber[n] = descr


z(0, '1', 'Piston')   
z(1, 'r*cos(theta)', 'Tilt x')
z(2, 'r*sin(theta)', 'Tilt y')
z(3, '-1 + 2*r**2', 'Defocus')
z(4, 'r**2*cos(2*theta)', 'Astig x')
z(5, 'r**2*sin(2*theta)', 'Astig y')
z(6, 'r*(-2 + 3*r**2)*cos(theta)', 'Coma x')
z(7, 'r*(-2 + 3*r**2)*sin(theta)', 'Coma y')
z(8, '1 - 6*r**2 + 6*r**4', 'Spherical')
z(9, 'r**3*cos(3*theta)', 'Trefoil x')
z(10, 'r**3*sin(3*theta)', 'Trefoil y')
z(11, 'r**2*(-3 + 4*r**2)*cos(2*theta)', 'Astig2 x')
z(12, 'r**2*(-3 + 4*r**2)*sin(2*theta)', 'Astig2 y')
z(13, 'r*(3 - 12*r**2 + 10*r**4)*cos(theta)', 'Coma2 x')
z(14, 'r*(3 - 12*r**2 + 10*r**4)*sin(theta)', 'Coma2 y')
z(15, '-1 + 12*r**2 - 30*r**4 + 20*r**6', 'Spherical2')
z(16, 'r**4*cos(4*theta)', 'Tetrafoil x')
z(17, 'r**4*sin(4*theta)', 'Tetrafoil y')
z(18, 'r**3*(-4 + 5*r**2)*cos(3*theta)', 'Trefoil2 x')
z(19, 'r**3*(-4 + 5*r**2)*sin(3*theta)', 'Trefoil2 y')
z(20, 'r**2*(6 - 20*r**2 + 15*r**4)*cos(2*theta)', 'Astig3 x')
z(21, 'r**2*(6 - 20*r**2 + 15*r**4)*sin(2*theta)', 'Astig3 y')
z(22, 'r*(-4 + 30*r**2 - 60*r**4 + 35*r**6)*cos(theta)', 'Coma3 x')
z(23, 'r*(-4 + 30*r**2 - 60*r**4 + 35*r**6)*sin(theta)', 'Coma3 y')
z(24, '1 - 20*r**2 + 90*r**4 - 140*r**6 + 70*r**8', 'Spherical3')
z(25, 'r**5*cos(5*theta)', 'Pentafoil x')
z(26, 'r**5*sin(5*theta)', 'Pentafoil y')
z(27, 'r**4*(-5 +6*r**2)*cos(4*theta)', 'Tetrafoil2 x')
z(28, 'r**4*(-5 +6*r**2)*sin(4*theta)', 'Tetrafoil2 x')
z(29, 'r**3*(10-30*r**2 + 21*r**4)*cos(3*theta)', 'Trefoil3 x')
z(30, 'r**3*(10-30*r**2 + 21*r**4)*sin(3*theta)', 'Trefoil3 y')
z(31, 'r**2*(-10 + 60*r**2 -105*r**4 + 56*r**6)*cos(2*theta)', 'Astig4 x')
z(32, 'r**2*(-10 + 60*r**2 -105*r**4 + 56*r**6)*sin(2*theta)', 'Astig4 y')
z(33, 'r*(5 - 60*r**2 + 210*r**4 - 280*r**6 + 126*r**8)*cos(theta)', 'Coma4 x')
z(34, 'r*(5 - 60*r**2 + 210*r**4 - 280*r**6 + 126*r**8)*sin(theta)', 'Coma4 y')
z(35, '-1 + 30*r**2 - 210*r**4 + 560*r**6 - 630*r**8 + 252*r**10', 'Spherical4')

def zernike(key, r, theta):
    if isreal(key):
        return zByNumber[key](r, theta)
    else:
        return zByName[key](r, theta)
        
def zernikeIm(key, shape):
    import numpy as np
    x, y = np.ogrid[:shape[0], :shape[1]]
    x = 2* x.astype('f')/shape[0] - 1
    y = 2* y.astype('f')/shape[1] - 1
    
    x = x - x.mean()
    y = y - y.mean()
    
    c = x + 1j*y
    
    r = np.abs(c)
    theta = np.angle(c)
    
    return zernike(key, r, theta)*(r<1)
    
    
def projectZ(image, key, weights = 1.0):
    from scipy.linalg import lstsq
    import numpy
    
    bf = zernikeIm(key, image.shape)
    
    return lstsq((bf*weights)[~numpy.isnan(image)].ravel().reshape([-1, 1]), (image*weights)[~numpy.isnan(image)].ravel())


def projectZ_rays(r, theta, pathlengths, key):
    from scipy.linalg import lstsq
    import numpy
    
    bf = zernike(key, r, theta) + 0*r
    
    #print bf
    return lstsq(bf.ravel().reshape([-1, 1]),
                 pathlengths.ravel())
    
    
def calcCoeffs(image, maxN, weights=1.0):
    im = image
    coeffs = []
    ress = []
    for n in range(maxN):
        c, res, rand, sing = projectZ(im, n, weights)
        print(('%d\t%s: %3.2f   residual=%3.2f' % (n, NameByNumber[n], c, res)))
        coeffs.append(c[0])
        ress.append(res)
        
        im = im - c*zernikeIm(n, im.shape)
        
    return coeffs, ress, im


def calcCoeffs_rays(r, theta, pathlengths, maxN, quiet=True):
    p = pathlengths
    coeffs = []
    ress = []
    for n in range(maxN):
        c, res, rand, sing = projectZ_rays(r, theta, p, n)
        if not quiet:
            print(('%d\t%s: %3.2f   residual=%3.2f' % (n, NameByNumber[n], c, res)))
        coeffs.append(c[0])
        ress.append(res)
        
        p = p - c * zernike(n, r, theta)
    
    return coeffs, ress, p