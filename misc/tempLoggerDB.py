#!/usr/bin/python
from PYME.Acquire.Hardware.DigiData.DigiDataClient import getDDClient
import time
import tempDB
import subprocess

#check to see if we're already running
ps = subprocess.Popen(['ps', '-e'], stdout=subprocess.PIPE).communicate()[0]

if ps.count('tempLoggerDB.py') == 1:

    dd = getDDClient()

    #f = open('/home/david/tempLog_10s.txt', 'a')
    

    while True:
        v = dd.GetAIValue(1)*1000./2.**15 - 273.15
        v1 = dd.GetAIValue(2)*1000./2.**15 - 273.15
        v2 = dd.GetAIValue(3)*1000./2.**15 - 273.15
        #f.write('%3.2f\t%3.2f\n' % (time.time(), v))
        #f.flush()
	t = time.time()
        tempDB.addEntry(t, v)
        tempDB.addEntry(t+.01, v1, 2)
        tempDB.addEntry(t + .02, v2, 3)

        time.sleep(10)

#else:
#    print 'logger already running - exiting'
