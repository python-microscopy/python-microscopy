#!/usr/bin/python

##################
# launchOnlyWorkers.py
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

import os
import subprocess
import sys
import time

def cpuCount():
    '''
    Returns the number of CPUs in the system
    borrowed from the python 'processing' package
    '''
    if sys.platform == 'win32':
        try:
            num = int(os.environ['NUMBER_OF_PROCESSORS'])
        except (ValueError, KeyError):
            num = 0
    elif sys.platform == 'darwin':
        try:
            num = int(os.popen('sysctl -n hw.ncpu').read())
        except ValueError:
            num = 0
    else: #assuming unix
        try:
            num = os.sysconf('SC_NPROCESSORS_ONLN')
        except (ValueError, OSError, AttributeError):
            num = 0
        
    if num >= 1:
        return num
    else:
        raise NotImplementedError('cannot determine number of cpus')


#get rid of any previously started queues etc...
#os.system('killall taskServerM.py')
#os.system('killall taskWorkerM.py')
#os.system('killall fitMon.py')

#launch pyro name server
#os.system('pyro-nsd start')


#get number of processors 
numProcessors = cpuCount()

if len(sys.argv) > 1:
        numProcessors = int(sys.argv[1])

if sys.platform == 'win32':
    #subprocess.Popen('python ./taskServerM.py', shell=True)

    #time.sleep(3)

    #subprocess.Popen('python ./fitMonP.py', shell=True)

    for i in range(numProcessors):
        subprocess.Popen('python ./taskWorkerMP.py', shell=True)

else: #operating systems which can launch python scripts directly
    #get rid of any previously started queues etc...
    #os.system('killall taskServerM.py')
    os.system('killall taskWorkerMP.py')
    #os.system('killall fitMonP.py')

    #subprocess.Popen('./taskServerM.py', shell=True)

    #time.sleep(3)

    #subprocess.Popen('./fitMonP.py', shell=True)

    for i in range(numProcessors):
        subprocess.Popen('./taskWorkerMP.py', shell=True)
