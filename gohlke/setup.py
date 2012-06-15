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
    config = Configuration('gohlke', parent_package, top_path)

    config.add_extension('_tifffile',
        sources=['tifffile.c'],
        include_dirs = [get_numpy_include_dirs()],
	extra_compile_args = ['-O3', '-fno-exceptions'],
        extra_link_args=['-static-libgcc'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'tiff reading',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''

''',
          license = "BSD",
          **configuration(top_path='').todict()
          )
