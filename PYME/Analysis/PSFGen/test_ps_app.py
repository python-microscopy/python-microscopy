#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
#
##################


import numpy as np
from PYME.Analysis.PSFGen import ps_app


def test_ps_app():
    X = np.arange(-5e3, 5e3, 70)
    P = np.arange(0, 1, .1)
    ps = ps_app.genWidefieldPSF(X, X, X, P)
    print(ps)
    
if __name__ == '__main__':
    test_ps_app()