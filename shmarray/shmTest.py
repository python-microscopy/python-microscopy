import numpy.random
import multiprocessing
import multiprocessing.sharedctypes
import shmarray
import numpy


data = shmarray.zeros(100)
#data = numpy.zeros(100)
#data = multiprocessing.sharedctypes.RawArray('d', 100)

d = None

def doFuzz(inds):
    #data, inds = args

    numpy.random.seed()
    r = numpy.random.random()
    #print multiprocessing.current_process()
    
    d[inds] = multiprocessing.current_process().pid

    return inds

def initFuzz(data):
    global d
    d = data

#data = shmarray.zeros(100)
#data = numpy.zeros(100)
#data = multiprocessing.sharedctypes.RawArray('d', 100)


pool = multiprocessing.Pool(4, initFuzz, (data,))

i = pool.map(doFuzz, range(0, 100), chunksize=10)

pool.close()

def foo(data):
    data[50:]=-1

p = multiprocessing.Process(target=foo, args=(data,))
p.start()
p.join()

print data
