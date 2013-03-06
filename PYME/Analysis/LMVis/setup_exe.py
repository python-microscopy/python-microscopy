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

ext_modules = [Extension("edgeDB", ["edgeDB.pyx"])]

import sys

if __name__ == '__main__':
    if sys.platform == 'win32':
        #from distutils.core import setup
        #try:
            import py2exe
            import os
            #import shutil
            import matplotlib
            setup(console=['VisGUI.py'],
              options={'py2exe':{'excludes':['pyreadline', 'Tkconstants','Tkinter','tcl', '_imagingtk','PIL._imagingtk', 'ImageTK', 'PIL.ImageTK', 'FixTk'],
                                 'includes':['OpenGL.platform.win32', 'OpenGL.arrays.*','scipy.ndimage', 'PYME.DSView.modules.*'],
                                 'dll_excludes': ['MSVCP90.dll', 'libzmq.dll'], 'optimize':0}},
              data_files=matplotlib.get_py2exe_datafiles(),
              #package_data = {'PYME':['DSView/icons/*']},
              scripts=['VisGUI.py']
              #cmdclass = {'build_ext': build_ext},
              #ext_modules = ext_modules
              )
        #except:
        #    setup()

    else:
        #try:
        from cx_Freeze import setup, Executable
        import matplotlib
        setup(executables=[Executable('VisGUI.py')],
            options= {'build_exe' : {
              'excludes' : ['pyreadline', 'Tkconstants', 'Tkinter', 'tcl', '_imagingtk', 'PIL._imagingtk', 'ImageTK', 'PIL.ImageTK', 'FixTk'],
              'packages' : ['OpenGL', 'OpenGL.platform', 'OpenGL.arrays']}},

            #data_files=matplotlib.get_py2exe_datafiles(),
          #cmdclass = {'build_ext': build_ext},
          #ext_modules = ext_modules
          )

        #except:
        #    setup(
              #cmdclass = {'build_ext': build_ext},
              #ext_modules = ext_modules
        #    )

