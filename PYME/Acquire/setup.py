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

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('Acquire', parent_package, top_path)
    
    config.add_subpackage('Hardware')
    config.add_subpackage('Protocols')
    config.add_data_dir('Scripts')
    config.add_subpackage('ui')
    config.add_subpackage('Utils')
    config.add_subpackage('webui')
    config.add_subpackage('htsms')
    #config.add_data_files('logo.png')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'Microscope control',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = """
Provides the microscope control and image acquisition components of PYME (what was previously PySMI)
""",
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
