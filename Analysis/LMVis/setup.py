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
from distutils.core import setup
from distutils.extension import Extension
#from Cython.Distutils import build_ext

ext_modules = [Extension("edgeDB", ["edgeDB.pyx"])]

import sys
if sys.platform == 'win32':
    #from distutils.core import setup
    try:
        import py2exe
        import os
        #import shutil
        import matplotlib
        setup(console=['VisGUI.py'],
          options={'py2exe':{'excludes':['pyreadline', 'Tkconstants','Tkinter','tcl', '_imagingtk','PIL._imagingtk', 'ImageTK', 'PIL.ImageTK', 'FixTk'], 'includes':['OpenGL.platform.win32'], 'optimize':0}},
          data_files=matplotlib.get_py2exe_datafiles(),
          #cmdclass = {'build_ext': build_ext},
          #ext_modules = ext_modules
          )
    except:
        setup()

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

