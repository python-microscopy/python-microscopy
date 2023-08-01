import time
import threading
import sys
import os
from collections import deque
import numpy as np
#from six.moves import xrange
#try:
#    from resource import getrusage, RUSAGE_SELF
#except ImportError:
#    RUSAGE_SELF = 0
#    def getrusage(who=0):
#        return [0.0, 0.0] # on non-UNIX platforms cpu_time always 0.0

p_stats = None
p_start_time = None
files = []



def profiler(frame, event, arg):
    if event not in ('call','return'): return profiler
    #### gather stats ####
    #rusage = getrusage(RUSAGE_SELF)
    #t_cpu = rusage[0] + rusage[1] # user time + system time
    t_cpu = time.time()
    code = frame.f_code 
    fn = frame.f_code.co_filename.split(os.sep)[-1]
    fun = (code.co_name, fn, code.co_firstlineno)
    
    if (not fn in files) and not (frame.f_back and (frame.f_back.f_code.co_filename.split(os.sep)[-1] in files)):
        return profiler
    #### get stack with functions entry stats ####
    ct = threading.currentThread()
    try:
        p_stack = ct.p_stack
    except AttributeError:
        ct.p_stack = deque()
        p_stack = ct.p_stack
    #### handle call and return ####
    
    if event == 'call':
        p_stack.append((t_cpu, fun))
    elif event == 'return':
        try:
            t_cpu_prev,f = p_stack.pop()
            assert f == fun
        except IndexError: # TODO investigate
            t_cpu_prev,f = p_start_time, None
        #call_cnt, t_sum, t_cpu_sum = p_stats.get(fun, (0, 0.0, 0.0))
        p_stats.append((fun, ct.name, t_cpu_prev, t_cpu-t_cpu_prev, len(p_stack)))
    return profiler


def profile_on(*filenames):
    global p_stats, p_start_time, files
    p_stats = []
    p_start_time = time.time()
    files = filenames
    threading.setprofile(profiler)
    sys.setprofile(profiler)


def profile_off():
    threading.setprofile(None)
    sys.setprofile(None)

def get_profile_stats():
    """
    returns dict[function_tuple] -> stats_tuple
    where
      function_tuple = (function_name, filename, lineno)
      stats_tuple = (call_cnt, real_time, cpu_time)
    """
    return p_stats

#### EXAMPLE ##################################################################

from time import sleep
from threading import Thread
import random



def t1(depth = 0):
    time.sleep(random.random()/10)
    if depth > 0:
        t1(depth-1)
        
def test_function():
    t1(3)    
    #pass

class T(Thread):
    def __init__(self):
        Thread.__init__(self)
    def run(self):                  # takes about 5 seconds
        for i in range(10):
            self.test_method()
            test_function()
    def test_method(self):
        sleep(random.random() / 10)

#profile_on('tProfile.py')
#######################
def runTests():
    threads = [T() for i in range(3)]
    for t in threads:
        t.start()
    for i in range(10):
        test_function()
    for t in threads:
        t.join()
        
runTests()


        
#t1(4)

#######################
#profile_off()

def plotProfile(s):
    #from pylab import *
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    
    a_s = np.array(s)
    
    tnames = list(set(a_s[:,1]))
    #print tnames
    
    z2 = np.zeros(2)
    
    ax = plt.gca()
    
    y0 = 1
    patches = []
    tb = a_s[0, 2]
    tbe = a_s[0, 2]
    for n, tn in enumerate(tnames):
        t_s = a_s[a_s[:,1]==tn, :]
        dm = t_s[:,4].max()
        for fcn, th, ts, te,d in t_s:
            y = d + y0
            #plot(np.array([0, te])+ts, z2 + y, lw=5)
            patches.append(mpatches.Rectangle((ts, y), te, 1))
            plt.text(ts, y + .6*np.random.rand(), fcn[0])
            tb = min(tb, ts)
            tbe = max(tbe, ts+te)
        y0 += dm + 2
        
    colors = (np.arange(len(patches))%20.)/20.
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    plt.show()
    plt.ylim(0, y0)
    plt.xlim(tb, tbe)

#from pprint import pprint
#pprint(get_profile_stats())
#plotProfile(get_profile_stats())