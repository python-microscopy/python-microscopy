#!/usr/bin/python

##################
# activeTest.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#scope.pa.start()
time.sleep(0.1)
l488.TurnOn()
time.sleep(1)
l405.TurnOn()
time.sleep(0.01)
l405.TurnOff()
time.sleep(1)
l405.TurnOn()
time.sleep(0.02)
l405.TurnOff()
time.sleep(1)
l405.TurnOn()
time.sleep(0.05)
l405.TurnOff()
time.sleep(1)
l405.TurnOn()
time.sleep(0.1)
l405.TurnOff()
time.sleep(1)
l405.TurnOn()
time.sleep(0.2)
l405.TurnOff()
time.sleep(1)
l405.TurnOn()
time.sleep(0.5)
l405.TurnOff()
time.sleep(1)
l405.TurnOn()
time.sleep(1)
l405.TurnOff()
time.sleep(1)
l405.TurnOn()
time.sleep(0.01)
l405.TurnOff()
time.sleep(1)
l488.TurnOff()
scope.pa.stop()
