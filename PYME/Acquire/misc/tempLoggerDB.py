#!/usr/bin/python

###############
# tempLoggerDB.py
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
from PYME.Acquire.Hardware.DigiData.DigiDataClient import getDDClient
import time
from PYME.misc import tempDB
import subprocess

if __name__ == '__main__':
    #check to see if we're already running
    ps = subprocess.Popen(['ps', '-e'], stdout=subprocess.PIPE).communicate()[0]
    
    if ps.count('tempLoggerDB.py') == 1:
    
        dd = getDDClient()
    
        #f = open('/home/david/tempLog_10s.txt', 'a')
        
    
        while True:
            v = dd.GetAIValue(1)*1000./2.**15 - 273.15
            v1 = dd.GetAIValue(2)*1000./2.**15 - 273.15 - 1.48 #correction for difference between sensors
            v2 = dd.GetAIValue(3)*1000./2.**15 - 273.15 - 0.20
            #f.write('%3.2f\t%3.2f\n' % (time.time(), v))
            #f.flush()
            t = time.time()
            tempDB.addEntry(t, v)
            tempDB.addEntry(t+.01, v1, 2)
            tempDB.addEntry(t + .02, v2, 3)
    
            time.sleep(10)
    
    #else:
    #    print 'logger already running - exiting'
