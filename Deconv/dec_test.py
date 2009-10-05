#!/usr/bin/python

##################
# dec_test.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import dec
from scipy import *
import profile
(x,y,z) = mgrid[-32:31,-32:31,-128:127]
g = exp(-(x**2 + y**2 + z**2))
h = exp(-(x**2 + y**2 + (z/3.0)**2))

d4 = dec.dec_4pi_c()
d4.psf_calc(h,1,shape(g))
d4.prepare()

profile.run("f = d4.deconv(ravel(cast['f'](g)), 1, alpha=g)")

#f = reshape(d4.Afunc(ravel(g)), shape(g))

#gplt.surf(f(:,5,:))

#f2 = d4.deconv(ravel(cast['f'](f)), 1, alpha=g)

#figure
#gplt.surf(f2(:,5,:))