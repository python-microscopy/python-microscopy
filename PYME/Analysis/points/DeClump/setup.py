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
if sys.platform == 'darwin':#MacOS
    linkArgs = ['-headerpad_max_install_names']
else:
    linkArgs = ['-static-libgcc']

def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('DeClump', parent_package, top_path)

    config.add_extension('deClump',
        sources=['deClump.c'],
        include_dirs = [get_numpy_include_dirs()],
	extra_compile_args = ['-O3', '-fno-exceptions'], # '-march=native', '-mtune=native'],
        extra_link_args=linkArgs)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(description = 'c coded pairwise distances',
    	author = 'David Baddeley',
       	author_email = 'd.baddeley@auckland.ac.nz',
       	url = '',
       	long_description = """
Provides a c-funtion for the fast & memory efficeint computation of a histogram of pairwise distances
""",
          license = "Proprietary",
          **configuration(top_path='').todict()
          )
