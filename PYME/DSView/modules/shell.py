#!/usr/bin/python
##################
# shell.py
#
# Copyright David Baddeley, 2011
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

import wx.py.shell
from PYME import config

def Plug(dsviewer):
    sh = wx.py.shell.Shell(id=-1,
                           parent=dsviewer, pos=wx.Point(0, 0), size=wx.Size(618, 451), style=0, locals=dsviewer.__dict__,
                           startupScript=config.get('dh5View-console-startup-file', None),
              introText='PYMEImage - note that help, license, etc. below is for Python, not PYME\n\n')

    sh.Execute('from pylab import *')
    sh.Execute('from scipy import ndimage')
    sh.Execute('from PYME.DSView import View3D, ViewIm3D')

    dsviewer.AddPage(page=sh, select=False, caption='Shell')

    dsviewer.sh = sh
