#!/usr/bin/env python

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('DSView', parent_package, top_path)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'Viewers for kdf and hdf',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''
Provides viewers for PYME's internal representation, kdf, and PYME hdf5 files
''',
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
