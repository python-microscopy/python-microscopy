#!/usr/bin/python

###############
# mProfile.py
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
#!/usr/bin/python
""" mProfile.py - Matlab(TM) style line based profiling for Python

Copyright: David Baddeley 2008
	   david_baddeley <at> yahoo.com.au

useage is similar to profiling in matlab (profile on, profile off, report),
with the major difference being that you have to specify the filenames in which
you want to do the profiling (to improve performance & to save wading through
lots of standard library code etc ...).

e.g.

mProfile.profileOn(['onefile.py', 'anotherfile.py'])

stuff to be profiled ....

mProfile.profileOff()
mProfile.report()

Due to the fact that we're doing this in python, and hooking every line, 
there is a substantial performance hit, although for the numeric code I wrote it
for (lots of vectorised numpy/scipy stuff) it's only on the order of ~30%.

Licensing: Take your pick of BSD or GPL
"""

import sys
import time
import os
from . import colorize_db_t
import webbrowser
import tempfile
import threading

if sys.version_info[0] >= 3:
    time_fcn = time.perf_counter
else:
    time_fcn = time.clock

#lStore.tPrev = time.time()

#lStore.lPrev = None
class mydictn(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self,key):
        return self.get(key, None)
        # if self.has_key(key):
        #     return dict.__getitem__(self, key)
        # else:
        #     return None


lStore = threading.local()
#lStore.tPrev = {}
#lStore.lPrev = mydictn()

filenames = []
files = {}
linecounts = {}
fullfilenames = {}

class mydict(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self,key):
        return self.get(key, 0)
        # if self.has_key(key):
        #     return dict.__getitem__(self, key)
        # else:
        #     return 0

def profileOn(fnames):
    global filenames, files, linecounts

    filenames = fnames
    
    files = {}
    linecounts = {}
    for f in fnames:
        files[f] = mydict()
        linecounts[f] = mydict()
        

    #lStore.tPrev = time.time()
    sys.settrace(te)
    threading.settrace(te)

def profileOff():
    sys.settrace(None)
    threading.settrace(None)

def te(frame, event, arg):
    #global filenames, files
    if not hasattr(lStore, 'tPrev'):
        lStore.tPrev = {}
    if not hasattr(lStore, 'lPrev'):
        lStore.lPrev = mydictn()
    
    fn = frame.f_code.co_filename.split(os.sep)[-1]
    funcName = fn + ' ' + frame.f_code.co_name 
    #print fn
    if fn in filenames and not lStore.lPrev[funcName] is None:
        t = time_fcn()
        files[lStore.lPrev[funcName][0]][lStore.lPrev[funcName][1]] += (t - lStore.tPrev[funcName])
        lStore.lPrev[funcName] = None
    
    if event == 'call':
        return te
    
    if event == 'line':
        if fn in filenames:
            linecounts[fn][frame.f_lineno] += 1
            fullfilenames[fn] = frame.f_code.co_filename
            lStore.lPrev[funcName] = (fn,frame.f_lineno)
            lStore.tPrev[funcName] = time_fcn()



def report(display=True, profiledir=None):
    if profiledir is None:
        tpath = os.path.join(tempfile.gettempdir(), 'mProf')
    else:
        tpath = profiledir
        
    if not os.path.exists(tpath):
        os.makedirs(tpath)

    for f in filenames:
        try:
            tfn = os.path.join(tpath,  f + '.html')
            colorize_db_t.colorize_file(files[f], linecounts[f], fullfilenames[f],open(tfn, 'w'))
            
            if display:
                webbrowser.open('file://' + tfn, 2)
        except KeyError:
            pass
        
