#!/usr/bin/python

##################
# setup.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

#!/usr/bin/env python
import sys

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration #, get_numpy_include_dirs
    config = Configuration('DSView', parent_package, top_path)
    #config.add_data_dir('icons')
    config.add_data_dir('modules/templates')
    config.add_subpackage('LUT')
    config.add_subpackage('modules')

    #if sys.platform == 'win32':
    #    config.add_scripts(['dh5view.cmd'])
    #else:
    #    config.add_scripts(['dh5view.py'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'Viewers for kdf and hdf',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = """
Provides viewers for PYME's internal representation, kdf, and PYME hdf5 files
""",
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
