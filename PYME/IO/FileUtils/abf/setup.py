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
def getBoostInclude():
    #import sys
    import os

    if 'BOOSTINCLUDE' in os.environ:
        return [os.environ['BOOSTINCLUDE']]
    else: #assume that we're on a platform (i.e. linux) where boost will be on the standard include path
        return []

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from glob import glob

    config = Configuration('abf', parent_package, top_path)

    config.add_extension('abf',
        sources=['abf.cpp', 'axon/AxAbfFio32/*.cpp', 'axon/Common/*.cpp', 'axon/Common/*.c'],
        include_dirs = [get_numpy_include_dirs()] + getBoostInclude(),
	extra_compile_args = ['-O3', '-fno-exceptions', '-D__UNIX__','-D__STF__'])
    
    return config

if __name__ == '__main__': 
    from numpy.distutils.core import setup
    setup(description = 'Axon abf file import',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = """ """,
          license = "GPL",
          **configuration(top_path='').todict()
          )
