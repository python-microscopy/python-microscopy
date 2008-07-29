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

def checkFilename(filename):
    #try and find filename in our script directories
    if os.path.exists(filename):
        return filename
    
    for p in execPath:
        fnp = os.path.join(p, filename)
        if os.path.exists(fnp):
            return fnp

    return filename #give up and let exec throw its normal error message


def _exec(codeObj, localVars, globalVars):
    exec codeObj in localVars,globalVars

def execBG(codeObj, localVars, globalVars):
    threading.Thread(target=_exec, args = (codeObj, localVars, globalVars)).start()
def execFile(filename, localVars, globalVars):    
    #fid = open(checkFilename(filename))
    #code = fid.read()
    #fid.close()

    execfile(checkFilename(filename), localVars, globalVars)

def execFileBG(filename, localVars, globalVars):
    #fid = open(checkFilename(filename))
    #code = fid.read()
    #fid.close()

    #execBG(checkFilename(filename), localVars, globalVars)
    threading.Thread(target=execfile, args = (checkFilename(filename), localVars, globalVars)).start()
