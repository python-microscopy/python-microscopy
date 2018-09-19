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
from distutils.core import setup
from distutils.extension import Extension
#from Cython.Distutils import build_ext

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('LMVis',parent_package,top_path)
    config.add_subpackage('Extras')
    config.add_subpackage('layers')
    config.add_subpackage('shader_programs')
    config.add_data_dir('shader_programs/shaders')
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(data_files = [], **configuration(top_path='').todict())