#!/usr/bin/python

##################
# relaxTest3.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#scope.pa.start()

#frs.OnBStartSpoolButton(None)

time.sleep(0.1)
l488.TurnOn()
time.sleep(1)

onTime = 5
offTimes = [0.2,0.5,1, 2,5, 10, 20, 50, 20, 10, 5, 2, 0.5, 0,2]

for offT in offTimes:
    l488.TurnOff()
    time.sleep(offT)
    l488.TurnOn()
    time.sleep(onTime)


l488.TurnOff()
scope.pa.stop()

#frs.OnBStopSpoolingButton(None)
