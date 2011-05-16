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
import sys

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('DSView', parent_package, top_path)
    config.add_data_dir('icons')

    if sys.platform == 'win32':
        config.add_scripts(['dh5view.cmd'])
    else:
        config.add_scripts(['dh5view.py'])

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
