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

#!/usr/bin/env python

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('ChemDE', parent_package, top_path)

    config.add_extension('StateMC',
        sources=['StateMonteCarlo.c'],
        include_dirs = [get_numpy_include_dirs()],
	extra_compile_args = ['-O3', '-fno-exceptions'])

    config.add_extension('countEvents',
        sources=['countEvents.c'],
        include_dirs = [get_numpy_include_dirs()],
	extra_compile_args = ['-O3', '-fno-exceptions'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'c coded pairwise distances',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''
Provides a c-funtion for the fast & memory efficeint computation of a histogram of pairwise distances
''',
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
