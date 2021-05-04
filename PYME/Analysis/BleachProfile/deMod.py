#!/usr/bin/python

##################
# deMod.py
#
# Copyright David Baddeley, 2009
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
from scipy.integrate import odeint

def flF2(N,t, argS):
    Na,Nd, Nd2 = N
    Ab,Aad,Ada, Aad2, Ada2,I = argS
    dNa = -(Ab + Aad + Aad2)*I*Na + Ada*Nd
    dNd = Aad*I*Na - Ada*Nd + Ada2*Nd2
    dNd2 = Aad2*Nd - Ada2*Nd2
    return np.array([dNa, dNd, dNd2])


def flF2Mod(p, t):
    Na= p[0]
    b = p[1]
    Nd = 0
    Nd2 = 0
    Ab,Aad,Ada, Aad2, Ada2 = p[2:]
    I = 1
    a = odeint(flF2, [Na, Nd, Nd2], t, ((Ab,Aad,Ada, Aad2, Ada2,I),))
    return a[:,0] + b


def flFsq(N,t, argS, Ia):
    Na,Nd = N
    Ab,Aad,Ada = argS
    I = 0

    ind = np.floor(t/0.007)
    #print len(Ia)
    if ind < len(Ia):
        I = Ia[ind]
        #print ind
    dNa = -(Ab + Aad)*I*Na**2 + Ada*Nd
    dNd = Aad*I*Na**2 - Ada*Nd
    #print dNa
    return np.array([dNa, dNd])


def flFsq1(N,t, argS):
    Na,Nd = N
    Ab,Aad,Ada,I = argS

    dNa = -(Ab + Aad)*I*Na**2 + Ada*Nd
    dNd = Aad*I*Na**2 - Ada*Nd
    #print dNa
    return np.array([dNa, dNd])


def flFpow(N,t, argS):
    Na,Nd = N
    Ab,Aad,Ada,I, pow = argS

    dNa = - Aad*I*Na**pow -Ab*I*Na + Ada*Nd
    dNd = Aad*I*Na**pow - Ada*Nd
    #print dNa
    return np.array([dNa, dNd])


#Model with thiol binding
def dBdt(B, t, kasc, kdis, kod, kdo, kbl, th, I):
    dB0 = - kasc*B[0]*th + kdis*B[1]  - kbl*I(t)*B[0]
    dB1 = kasc*B[0]*th - kdis*B[1] - kod*I(t)*B[1] + kdo*B[2]
    dB2 = kod*I(t)*B[1] - kdo*B[2]
    return[dB0, dB1, dB2]

def thiolInt(t, B0, kasc, kdis, kod, kdo, kbl, th, I):
    return odeint(dBdt, B0, t, args = (kasc, kdis, kod, kdo, kbl, th, I))
