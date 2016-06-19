#!/usr/bin/python

##################
# fnlGen.py
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

import os
import wx

def genFnl(inDir):
    """Generates a list of files in inDir with full paths"""
    if not (inDir[-1] == os.sep):
        inDir += os.sep #append a / to directroy name if necessary

    fnl = os.listdir(inDir)
    return [inDir + f for f in fnl]

def GuiGetDirList(promptString='Select directory containing image files', defaultPath = ''):
    inDir = wx.DirSelector(promptString, defaultPath)
    if inDir == '':
        return []
    else:
        return genFnl(inDir)
