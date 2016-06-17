#!/usr/bin/python

##################
# snells.py
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
##################
import numpy as np

def theta_t(theta_i, ni, nt):
    return np.arcsin(np.minimum(ni*np.sin(theta_i)/nt, 1.0))
    
def Rs(theta_i, ni, nt):
    th_t = theta_t(theta_i, ni, nt)
    return ((ni*np.cos(theta_i) - nt*np.cos(th_t))/(ni*np.cos(theta_i) + nt*np.cos(th_t)))**2
    
def Ts(theta_i, ni, nt):
    return 1 - Rs(theta_i, ni, nt)
    
def Rp(theta_i, ni, nt):
    th_t = theta_t(theta_i, ni, nt)
    return ((ni*np.cos(th_t) - nt*np.cos(theta_i))/(ni*np.cos(th_t) + nt*np.cos(theta_i)))**2
    
def Tp(theta_i, ni, nt):
    return 1 - Rp(theta_i, ni, nt)
    
    
    
def ASF(nIm, nM, NA):
    """Calculate the forshortening for a given refractive index mismatch.
    Uses geometrical optics, rather than the paraxial approximation. Doesn't
    really belong here, but not really worth a new file either."""
    
    #our fluorophore emits uniformly, but we'll never see anything which 
    #goes in the wrong direction
    th_m = np.linspace(.01, np.pi/2)
    
    #the angle that the 'rays' from the fluorophore will be bent to at the 
    #refractive index boundary. ie the angle of the rays in the glass
    th_im = theta_t(th_m, nM, nIm)
    
    #the ratio of true to apparent focus as a function of angle    
    asf = np.tan(th_im)/np.tan(th_m)
    
    #how much will be transmitted?
    T = (Tp(th_m, nM, nIm) + Ts(th_m, nM, nIm))/2
    #we need to integrate round the ring at each angle
    w = T*2*np.pi*np.sin(th_m)
    
    #and limit to the detection objective NA
    r = nIm*np.sin(th_im)
    w *= r < NA
    
    w = w/w.sum()
    
    return (asf*w).sum()