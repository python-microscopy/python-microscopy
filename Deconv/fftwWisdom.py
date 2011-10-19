# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:12:10 2011

@author: dbad004
"""

import fftw3f
import os

WISDOMFILE = os.path.join(os.path.split(__file__)[0], 'fftw_wisdom')

def load_wisdom():
    if os.path.exists(WISDOMFILE):
        f = open(WISDOMFILE, 'r')
        fftw3f.import_wisdom_from_string(f.read())
        f.close()
    
def save_wisdom():
    f = open(WISDOMFILE, 'w')
    f.write(fftw3f.export_wisdom_to_string())
    f.close()