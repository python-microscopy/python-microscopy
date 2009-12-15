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

import sys
if sys.platform == 'win32'
    from distutils.core import setup
    import py2exe
    import os
    #import shutil
    import matplotlib
    setup(console=['VisGUI.py'],
          options={'py2exe':{'excludes':['pyreadline', 'Tkconstants','Tkinter','tcl', '_imagingtk','PIL._imagingtk', 'ImageTK', 'PIL.ImageTK', 'FixTk'], 'includes':['OpenGL.platform.win32'], 'optimize':0}},
          data_files=matplotlib.get_py2exe_datafiles())

