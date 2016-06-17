#!/usr/bin/python

##################
# fakeShutters.py
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

#!/usr/bin/python
""" Placeholder module for shutters, should they be reimplemented"""

CH1 = 1
CH2 = 2
CH3 = 4
CH4 = 8
ALL = 0xFF

def closeShutters(shutters):
    pass

def openShutters(shutters):
    pass

def getShutterStates():
    return 0xFF

def setShutterStates(states):
    pass
	
def getShutterState(shutter):
    return True

def setPort(port):
    pass

def init():
    pass
