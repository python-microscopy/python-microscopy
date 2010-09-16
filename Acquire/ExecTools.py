#!/usr/bin/python

##################
# ExecTools.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python
'''defines a few neat things to allow scripts to be executed in background & to
allow a user script directory'''
import threading
#import pythoncom

import os
import sys

homedir = os.path.expanduser('~') #unix & possibly others ...
execPath = []

if 'USERPROFILE' in os.environ.keys(): #windows
    homedir = os.environ['USERPROFILE']

localScriptPath = os.path.join(homedir, 'PYMEScripts')
if os.path.exists(localScriptPath):
    sys.path.append(localScriptPath)
    execPath.append(localScriptPath)

#append global script directory _after_ user script directory so we use local init.py
execPath.append(os.path.join(os.path.dirname(__file__), 'Scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

global defGlobals
global defLocals
defGlobals = {}
defLocals = {}

bgInitThreads = []
#bgInitStatus = {}

class HWNotPresent(Exception):
    pass


def setDefaultNamespace(locals, globals):
    global defGlobals
    global defLocals

    defLocals = locals
    defGlobals = globals

def checkFilename(filename):
    #try and find filename in our script directories
    if os.path.exists(filename):
        return filename
    
    for p in execPath:
        fnp = os.path.join(p, filename)
        if os.path.exists(fnp):
            return fnp

    return filename #give up and let exec throw its normal error message


def _exec(codeObj, localVars = None, globalVars = None):
    exec codeObj in localVars,globalVars

def execBG(codeObj, localVars = defLocals, globalVars = defGlobals):
    t = threading.Thread(target=_exec, args = (codeObj, localVars, globalVars))
    t.start()
    return t

def execFile(filename, localVars = defLocals, globalVars = defGlobals):
    #fid = open(checkFilename(filename))
    #code = fid.read()
    #fid.close()

    execfile(checkFilename(filename), localVars, globalVars)

def execFileBG(filename, localVars = defLocals, globalVars = defGlobals):
    #fid = open(checkFilename(filename))
    #code = fid.read()
    #fid.close()

    #execBG(checkFilename(filename), localVars, globalVars)
    threading.Thread(target=execfile, args = (checkFilename(filename), localVars, globalVars)).start()

def _bginit(name, codeObj):
    global defGlobals
    global defLocals
    _exec("splash.SetMessage('%s', 'Initialising %s ...')" % (name,name), defGlobals, defLocals)
    try:
        _exec(codeObj, defGlobals, defLocals)
        _exec("splash.SetMessage('%s', 'Initialising %s ... DONE')" % (name,name), defGlobals, defLocals)
    except HWNotPresent:
        _exec("splash.SetMessage('%s', 'Initialising %s ... NOT PRESENT')" % (name,name), defGlobals, defLocals)
    except Exception, e:
        _exec("splash.SetMessage('%s', 'Initialising %s ... FAIL')" % (name,name), defGlobals, defLocals)
        raise e


def InitBG(name, codeObj):
    t = threading.Thread(target=_bginit, args = (name,codeObj))
    t.start()
    bgInitThreads.append(t)
    return t
    

def joinBGInit():
    for t in bgInitThreads:
        print t
        t.join()

def InitGUI(code):
    defLocals['postInit'].append(code)
