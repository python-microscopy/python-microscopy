#scope.pa.start()

#frs.OnBStartSpoolButton(None)

time.sleep(0.1)
l488.TurnOn()
time.sleep(1)

offTime = 5
onTimes = [0.01,0.02,0.05,0.1,0,2,0.5,1]

for onT in onTimes:
    l405.TurnOn()
    time.sleep(onT)
    l405.TurnOff()
    time.sleep(offTime)


l488.TurnOff()
scope.pa.stop()

#frs.OnBStopSpoolingButton(None)
