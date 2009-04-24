#!/usr/bin/python
''' mProfile.py - Matlab(TM) style line based profiling for Python

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


TODO: fix hard coded temp directory so it also works on non *nix platforms

Licensing: Take your pick of BSD or GPL
'''

import sys
import time
import os
import colorize_db_t
import webbrowser
import tempfile
import threading

#tPrev = time.time()

#lPrev = None
class mydictn(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self,key):
        if self.has_key(key):
            return dict.__getitem__(self, key)
        else:
            return None

tPrev = {}
lPrev = mydictn()

filenames = []
files = {}
fullfilenames = {}

class mydict(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self,key):
        if self.has_key(key):
            return dict.__getitem__(self, key)
        else:
            return 0

def profileOn(fnames):
    global filenames, files, tPrev

    filenames = fnames
    
    files = {}
    for f in fnames:
        files[f] = mydict()

    #tPrev = time.time()
    sys.settrace(te)
    threading.settrace(te)

def profileOff():
    sys.settrace(None)
    treading.settrace(None)

def te(frame, event, arg):
    global tPrev, filenames, files, lPrev
    fn = frame.f_code.co_filename.split(os.sep)[-1]
    funcName = fn + ' ' + frame.f_code.co_name 
    #print fn
    if fn in filenames and not lPrev[funcName] == None:
        t = time.time()
        files[lPrev[funcName][0]][lPrev[funcName][1]] += (t - tPrev[funcName])
        lPrev[funcName] = None
    if event == 'call':
        return te
    if event == 'line':
        if fn in filenames:
            fullfilenames[fn] = frame.f_code.co_filename
            lPrev[funcName] = (fn,frame.f_lineno)
            tPrev[funcName] = time.time()



def report():
    tpath = os.path.join(tempfile.gettempdir(), 'mProf')
    if not os.path.exists(tpath):
        os.makedirs(tpath)

    for f in filenames:
        tfn = os.path.join(tpath,  f + '.html')
        colorize_db_t.colorize_file(files[f], fullfilenames[f],open(tfn, 'w'))
        webbrowser.open(tfn, 2)
        
