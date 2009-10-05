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
    config = Configuration('Hardware', parent_package, top_path)
    
    config.add_subpackage('AndorIXon')
    config.add_subpackage('DigiData')
    config.add_subpackage('Simulator')
    config.add_subpackage('Old')
    config.add_subpackage('Piezos')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'Hardware support',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = '''
Assorted hardware support
''',
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
