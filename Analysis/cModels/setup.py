#!/usr/bin/env python

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('cModels', parent_package, top_path)

    config.add_extension('gauss_app',
        sources=['gauss_ap.c'],
        include_dirs = [get_numpy_include_dirs()],
	extra_compile_args = ['-O3', '-fno-exceptions', '-march=nocona'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'model functions in c',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''
Provides a c-funtions to allow rapid computation of models for the purposes of e.g. fitting
''',
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
