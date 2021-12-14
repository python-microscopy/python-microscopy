#!/usr/bin/python

###############
# hillcurve.py
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

# from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from PYME.Analysis._fithelpers import *

def hill(p, ratio):
    """Modified Hills function for fitting to stress vs ratio data.
    
    Parameters (in  array p) are:
        p[0] = n 
        p[1] = Ka (EC50)
        p[2] = Fmax
        p[3] = Fmin
    
    """
    n, Ka, Fmax, Fmin = p 
    return Fmin + Fmax*(ratio**n)/(Ka**n + ratio**n)

if __name__ == '__main__':
    #load control data
    ratio_control, stress_control = np.loadtxt('control.csv').T
    plt.plot(ratio_control, stress_control, 'x', label='control')

    #do fit
    fit_results = FitModel(hill, [1, 2.7, 50, 1], ratio_control)
    p_control = fit_results[0]

    print(("""
    Control:
    ---------
    n = %3.2f
    EC50 = %3.2f
    Fmax = %3.2f
    Fmin = %3.2f

    """ % p_control))

    ratio = np.linspace(1.5, 4)
    plt.plot(ratio, hill(p_control, ratio))


    #load cptome data
    ratio_cptome, stress_cptome = np.loadtxt('cptome.csv').T
    plt.plot(ratio_cptome, stress_cptome, 'x', label='CPTOME')

    #do fit
    fit_results = FitModel(hill, [1, 2.7, 50, 1], ratio_cptome)
    p_cptome = fit_results[0]

    print(("""
    CPTOME:
    ---------
    n = %3.2f
    EC50 = %3.2f
    Fmax = %3.2f
    Fmin = %3.2f

    """ % p_cptome))

    plt.plot(ratio, hill(p_cptome, ratio))

    plt.xlabel('Ratio - 340/380')
    plt.ylabel('Stress')
    plt.legend()