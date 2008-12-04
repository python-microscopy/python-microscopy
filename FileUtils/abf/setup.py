#!/usr/bin/env python
def getBoostInclude():
    #import sys
    import os

    if 'BOOSTINCLUDE' in os.environ:
        return [os.environ['BOOSTINCLUDE']]
    else: #assume that we're on a platform (i.e. linux) where boost will be on the standard include path
        return []

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from glob import glob

    config = Configuration('abf', parent_package, top_path)

    config.add_extension('abf',
        sources=['abf.cpp', 'axon/AxAbfFio32/*.cpp', 'axon/Common/*.cpp', 'axon/Common/*.c'],
        include_dirs = [get_numpy_include_dirs()] + getBoostInclude(),
	extra_compile_args = ['-O3', '-fno-exceptions', '-D__UNIX__','-D__STF__'])
    
    return config

if __name__ == '__main__': 
    from numpy.distutils.core import setup
    setup(description = 'Axon abf file import',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = ''' ''',
          license = "GPL",
          **configuration(top_path='').todict()
          )
