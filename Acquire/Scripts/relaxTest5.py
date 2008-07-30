#scope.pa.start()

#frs.OnBStartSpoolButton(None)

time.sleep(0.1)
l488.TurnOn()
time.sleep(10)

onTime = 10
offTimes = 20*ones(10)

for offT in offTimes:
    l488.TurnOff()
    time.sleep(offT)
    l488.TurnOn()
    time.sleep(onTime)


l488.TurnOff()
scope.pa.stop()

#frs.OnBStopSpoolingButton(None)
