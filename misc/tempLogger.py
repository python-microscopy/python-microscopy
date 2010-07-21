from PYME.Acquire.Hardware.DigiData.DigiDataClient import getDDClient
import time

dd = getDDClient()

f = open('/home/david/tempLog_10s.txt', 'a')

while True:
    v = dd.GetAIValue(1)*1000./2.**15
    f.write('%3.2f\t%3.2f\n' % (time.time(), v))
    f.flush()
    time.sleep(10)
