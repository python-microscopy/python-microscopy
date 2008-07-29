import sys
import time
import os
import colorize_db_t

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
            lPrev[funcName] = (fn,frame.f_lineno)
            tPrev[funcName] = time.time()



def report():
    for f in filenames:
        colorize_db_t.colorize_file(files[f], f,open('/tmp/mProf/' + f + '.html', 'w'))
