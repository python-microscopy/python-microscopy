#!/usr/bin/python

##################
# setup.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from distutils.core import setup
from distutils.extension import Extension
#from Cython.Distutils import build_ext

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('LMVis',parent_package,top_path)
    config.add_subpackage('Extras')
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(data_files = [], **configuration(top_path='').todict())