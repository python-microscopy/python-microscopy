#!/usr/bin/env python

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from glob import glob

    config = Configuration('abf', parent_package, top_path)

    config.add_extension('abf',
        sources=['abf.cpp'] + glob('axon/AxAbfFio32/*.cpp') + glob('axon/Common/*.cpp') + glob('axon/Common/*.c'),
        include_dirs = [get_numpy_include_dirs()],
	extra_compile_args = ['-O3', '-fno-exceptions', '-D__UNIX__'])

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
