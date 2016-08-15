#!/usr/bin/python

##################
# statusLog.py
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

import random

statusTexts = {}
statusKeys = []

statusDispFcn = None


class StatusLogger:
    def __init__(self, initText=''):
        global statusKeys
        self.key = random.random()
        statusKeys.append(self.key)
        self.setStatus(initText)

    def __del__(self):
        global statusTexts, statusKeys
        statusTexts.pop(self.key)
        statusKeys.remove(self.key)

    def setStatus(self, statusText):
        global statusTexts
        statusTexts[self.key] = statusText
        if not statusDispFcn is None:
            statusDispFcn(GenStatusText())

def GenStatusText():    
    st = '\t'.join([statusTexts[key] for key in statusKeys])

    return st

def SetStatusDispFcn(dispFcn):
    global statusDispFcn
    statusDispFcn = dispFcn
