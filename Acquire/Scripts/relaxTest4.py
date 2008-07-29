#scope.pa.start()

#frs.OnBStartSpoolButton(None)

time.sleep(0.1)
l473.TurnOn()
time.sleep(1)

onTime = 5
offTimes = [0.2,0.5,1, 2,5, 10, 20, 50, 20, 10, 5, 2, 0.5, 1,0.2]

for offT in offTimes:
    l473.TurnOff()
    time.sleep(offT)
    l473.TurnOn()
    time.sleep(onTime)


l473.TurnOff()
scope.pa.stop()

#frs.OnBStopSpoolingButton(None)
