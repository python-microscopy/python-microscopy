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
import py2exe
import os
import shutil
import matplotlib

setup(console=['taskWorkerME.py'],
      options={'py2exe':{'excludes':['pyreadline', 'Tkconstants','Tkinter','tcl', '_imagingtk','PIL._imagingtk', 'ImageTK', 'PIL.ImageTK', 'FixTk'],
                         'includes':['pyscr'],
                         'dll_excludes':['MSVCP90.dll']}},
      data_files=matplotlib.get_py2exe_datafiles())
os.system('python buildScr.py PYMEScreensaver.py')
shutil.copyfile('Dist/PYMEScreensaver/PYMEScreensaver.scr', 'Dist/PYMEScreensaver.scr')
shutil.rmtree('Dist/PYMEScreensaver')
