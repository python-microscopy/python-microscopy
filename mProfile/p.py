#!/usr/bin/python

##################
# p.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import sys
import time
import os
import colorize_db_t
import webbrowser

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

def profOn(fnames):
    global filenames, files, tPrev

    filenames = fnames
    
    files = {}
    for f in fnames:
        files[f] = mydict()

    #tPrev = time.time()
    sys.settrace(te)

def profOff():
    sys.settrace(None)

def te(frame, event, arg):
    global tPrev, filenames, files, lPrev
    fn = frame.f_code.co_filename.split(os.sep)[-1]
    funcName = fn + ' ' + frame.f_code.co_name 
    #print funcName
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
    if not os.path.exists('/tmp/mProf'):
        os.makedirs('/tmp/mProf')

    for f in filenames:
        colorize_db_t.colorize_file(files[f], fullfilenames[f],open('/tmp/mProf/' + f + '.html', 'w'))
        webbrowser.open('/tmp/mProf/' + f + '.html', 2)
        
