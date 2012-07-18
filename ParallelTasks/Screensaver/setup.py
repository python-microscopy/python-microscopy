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
