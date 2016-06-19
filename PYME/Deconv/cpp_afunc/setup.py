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

from distutils.core import setup, Extension

module1 = Extension('cDec',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include'],
                    libraries = ['fftw3f'],
                    library_dirs = ['/usr/local/lib'],
                    sources = ['fft_test.c'])

setup (name = 'cDec',
       version = '1.0',
       description = 'c routines for deconvolution package',
       author = 'David Baddeley',
       author_email = 'baddeley@kip.uni-heidelberg.de',
       url = '',
       long_description = """
optimised routines for processor intensive parts of deconvolution
""",
       ext_modules = [module1])
