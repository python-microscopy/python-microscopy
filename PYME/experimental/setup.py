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
import os

if sys.platform == 'darwin':#MacOS
    linkArgs = []
else:
    linkArgs = ['-static-libgcc']

from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):

    from numpy.distutils.core import Extension
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    
    cur_dir = os.path.dirname(__file__)
    
    ext = Extension(name='.'.join([parent_package, 'experimental', '_octree']),
                    sources=[os.path.join(cur_dir, '_octree.pyx')],
                    include_dirs= get_numpy_include_dirs(), #+ extra_include_dirs,
                    extra_compile_args=['-O3', '-fno-exceptions', '-ffast-math', '-march=native', '-mtune=native'],
                    extra_link_args=linkArgs)

    ext2 = Extension(name='.'.join([parent_package, 'experimental', 'triangle_mesh_utils']),
                    sources=[os.path.join(cur_dir, 'mesh_utils.pyx'), os.path.join(cur_dir, 'triangle_mesh_utils.c')],
                    include_dirs= get_numpy_include_dirs(), #+ extra_include_dirs,
                    extra_compile_args=['-O3', '-fno-exceptions', '-ffast-math', '-march=native', '-mtune=native'],
                    extra_link_args=linkArgs)
    
    config = Configuration('experimental', parent_package, top_path, ext_modules=cythonize([ext, ext2]))
    
    # config = Configuration('pymecompress', parent_package, top_path)
    #
    # config.add_extension('bcl',
    #     sources=['bcl.pyx', 'src/huffman.c', 'quantize.c'],
    #     include_dirs = ['src', get_numpy_include_dirs()] + extra_include_dirs,
    # extra_compile_args = ['-O3', '-fno-exceptions', '-ffast-math', '-march=native', '-mtune=native'],
    #     extra_link_args=linkArgs)
    
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    
    setup(description='various experimental functions',
          author='David Baddeley',
          author_email='david.baddeley@yale.edu',
          url='',
          long_description="""
""",
          license="GPL",
          #cmdclass={'build_ext': build_ext},
          **configuration(top_path='').todict()
          )