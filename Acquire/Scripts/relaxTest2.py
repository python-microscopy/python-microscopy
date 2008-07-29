#scope.pa.start()

#frs.OnBStartSpoolButton(None)

time.sleep(0.1)
l488.TurnOn()
time.sleep(1)

onTime = 5
offTimes = [0.01,0.02,0.05,0.1,0,2,0.5,1, 5, 10]

for offT in offTimes:
    l488.TurnOff()
    time.sleep(offT)
    l488.TurnOn()
    time.sleep(onTime)


l488.TurnOff()
scope.pa.stop()

#frs.OnBStopSpoolingButton(None)
