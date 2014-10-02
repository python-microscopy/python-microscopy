#!/usr/bin/python

###############
# populateStats.py
#
# Copyright David Baddeley, 2012
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
################
#!/usr/bin/python
import sys
import os

#put the folder above this one on the python path
#print __file__
#print os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(os.path.split(os.path.abspath(__file__))[0])

#let Django know where to find the settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'SampleDB2.settings'

from samples.models import File
from samples import calcEventStats


def addStats():
    #dir_size = 0
    files = File.objects.all()
    for f in files:
        calcEventStats.getStats(f)


if __name__ == '__main__':
    addStats()
    
