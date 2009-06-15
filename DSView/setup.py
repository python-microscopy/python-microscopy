#!/usr/bin/env python
import sys

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('DSView', parent_package, top_path)

#    if sys.platform == 'win32':
#        config.add_scripts('dh5view.cmd')

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
          #data_files = ['dh5view.cmd']
          **configuration(top_path='').todict()
          )
