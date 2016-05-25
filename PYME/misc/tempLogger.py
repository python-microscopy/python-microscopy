#!/usr/bin/python

###############
# tempLogger.py
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
from PYME.Acquire.Hardware.DigiData.DigiDataClient import getDDClient
import time

if __name__ == '__main__':
    dd = getDDClient()
    
    f = open('/home/david/tempLog_10s.txt', 'a')
    
    while True:
        v = dd.GetAIValue(1)*1000./2.**15
        f.write('%3.2f\t%3.2f\n' % (time.time(), v))
        f.flush()
        time.sleep(10)
