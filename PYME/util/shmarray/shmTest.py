#!/usr/bin/python

###############
# shmTest.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################

def test():
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
    
    print(data)
    
if __name__ == '__main__':
    test()
