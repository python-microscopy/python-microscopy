#!/usr/bin/python

##################
# dec_test.py
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

__test__= False

if __name__ == '__main__':
    import dec
    import numpy as np
    import profile
    (x,y,z) = np.mgrid[-32:31,-32:31,-128:127]
    g = np.exp(-(x**2 + y**2 + z**2))
    h = np.exp(-(x**2 + y**2 + (z/3.0)**2))
    
    d4 = dec.dec_4pi_c()
    d4.psf_calc(h,1,np.shape(g))
    d4.prepare()
    
    profile.run("f = d4.deconv(ravel(cast['f'](g)), 1, alpha=g)")
    
    #f = reshape(d4.Afunc(ravel(g)), shape(g))
    
    #gplt.surf(f(:,5,:))
    
    #f2 = d4.deconv(ravel(cast['f'](f)), 1, alpha=g)
    
    #figure
    #gplt.surf(f2(:,5,:))