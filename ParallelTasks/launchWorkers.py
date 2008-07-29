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
    else:
        try:
            num = os.sysconf('SC_NPROCESSORS_ONLN')
        except (ValueError, OSError, AttributeError):
            num = 0
        
    if num >= 1:
        return num
    else:
        raise NotImplementedError, 'cannot determine number of cpus'


#get rid of any previously started queues etc...
os.system('killall taskServerM.py')
os.system('killall taskWorkerM.py')
os.system('killall fitMon.py')

#launch pyro name server
os.system('pyro-nsd start')


#get number of processors 
numProcessors = cpuCount()

subprocess.Popen('python ./taskServerM.py', shell=True)

time.sleep(1)

subprocess.Popen('python ./fitMon.py', shell=True)

for i in range(numProcessors):
    subprocess.Popen('python ./taskWorkerM.py', shell=True)
