#!/usr/bin/env python

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('Acquire', parent_package, top_path)
    
    config.add_subpackage('Hardware')
    config.add_data_dir('Scripts')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'Microscope control',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''
Provides the microscope control and image acquisition components of PYME (what was previously PySMI)
''',
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
