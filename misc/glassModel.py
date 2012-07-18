#!/usr/bin/python

##################
# glassModel.py
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

#constants taken from http://www.cvilaser.com/Common/PDFs/Dispersion_Equations.pdf
sellConstants = {
'MgF2 e' :         (4.13440230E-01,  5.04974990E-01,  2.49048620E+00,  1.35737865E-03, 8.23767167E-03, 5.65107755E+02),
'MgF2 o' :         (4.87551080E-01,  3.98750310E-01,  2.31203530E+00,  1.88217800E-03, 8.95188847E-03, 5.66135591E+02),
'Sapphire e' :     (1.50397590E+00,  5.50691410E-01,  6.59273790E+00,  5.48041129E-03, 1.47994281E-02, 4.02895140E+02),
'Sapphire o' :     (1.43134930E+00,  6.50547130E-01,  5.34140210E+00,  5.27992610E-03, 1.42382647E-02, 3.25017834E+02),
'CaF2' :           (5.67588800E-01,  4.71091400E-01,  3.84847230E+00,  2.52642999E-03, 1.00783328E-02, 1.20055597E+03),
'Fused Silica' :   (6.96166300E-01,  4.07942600E-01,  8.97479400E-01,  4.67914826E-03, 1.35120631E-02, 9.79340025E+01),
'Schott BK7' :     (1.03961212E+00,  2.31792344E-01,  1.01046945E+00,  6.00069867E-03, 2.00179144E-02, 1.03560653E+02),
'Schott N-BK7':    (1.03961212E+00,  2.31792344E-01,  1.01046945E+00,  6.00069867E-03, 2.00179144E-02, 1.03560653E+02),
'Schott F2':       (1.34533359E+00,  2.09073118E-01,  9.37357162E-01,  9.97743871E-03, 4.70450767E-02, 1.11886764E+02),
'Schott N-F2':     (1.39757037E+00,  1.59201403E-01,  1.26865430E+00,  9.95906143E-03, 5.46931752E-02, 1.19248346E+02),
'Schott SF2' :     (1.40301821E+00,  2.09073176E-01,  9.39056586E-01,  1.05795466E-02, 4.93226978E-02, 1.12405955E+02),
'Schott SF10' :    (1.61625977E+00,  2.59229334E-01,  1.07762317E+00,  1.27534559E-02, 5.81983954E-02, 1.16607680E+02),
'Schott N-SF10' :  (1.62153902E+00,  2.56287842E-01,  1.64447552E+00,  1.22241457E-02, 5.95736775E-02, 1.47468793E+02),
'Schott SF11' :    (1.73848403E+00,  3.11168974E-01,  1.17490871E+00,  1.36068604E-02, 6.15960463E-02, 1.21922711E+02),
'Schott N-SF11' :  (1.73759695E+00,  3.13747346E-01,  1.89878101E+00,  1.13188707E-02, 6.23068142E-02, 1.55236290E+02),
'Schott N-LAK21' : (1.22718116E+00,  4.20783743E-01,  1.01284843E+00,  6.02075682E-03, 1.96862889E-02, 8.84370099E+01)
}


def Sellmeier(lamb, B1, B2, B3, C1, C2, C3):
    L2 = (lamb/1e3)**2
    return np.sqrt(1 + (B1*L2)/(L2-C1) + (B2*L2)/(L2-C2) + (B3*L2)/(L2-C3))


cauchConstants = {
'Index Matching Fluid' : (1.502787, 455872.4e-8, 9.844856e-5)
}

def Cauchy(lamb, *args):
    L2 = (lamb/1e3)**2

    n = 0

    for a, n in zip(args, range(len(args))):
        n += a/(L2**n)

    return n
