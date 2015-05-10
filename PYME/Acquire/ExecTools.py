#!/usr/bin/python

##################
# ExecTools.py
#
# Copyright David Baddeley, 2009
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
##################

#!/usr/bin/python
'''Defines a few neat things to allow scripts to be executed in background during 
initialisation & to allow a user script directory'''
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
    '''An exception which, if thrown during initialisation, will result in a more
    informative error message than just failure'''
    pass


def setDefaultNamespace(locals, globals):
    '''Populate the namespace in which the initialisation script will run.
    
    Parameters
    ----------
    locals : dict
    globals : dict    
    '''
    global defGlobals
    global defLocals

    defLocals = locals
    defGlobals = globals

def checkFilename(filename):
    '''Check both the scripts directory in the PYME tree and a separate user 
    script directory, `~/PYMEScripts` for an initialisation script of the given
    name. 
    
    Returns
    -------
    filename : string
        The full path to the requested script
    '''
    #try and find filename in our script directories
    if os.path.exists(filename):
        return filename
    
    for p in execPath:
        fnp = os.path.join(p, filename)
        if os.path.exists(fnp):
            return fnp

    return filename #give up and let exec throw its normal error message

if sys.version_info.major == 2:
    def _exec(codeObj, localVars = None, globalVars = None):
        exec codeObj in localVars,globalVars
    def _execfile(filename, localVars=None, globalVars=None):
        execfile(filename, localVars, globalVars)
else: #Python 3
    def _exec(codeObj, localVars = None, globalVars = None):
        exec(codeObj,localVars,globalVars)
    def _execfile(filename, localVars=None, globalVars=None):
        exec(compile(open(filename).read(), filename, 'exec'), localVars, globalVars)

def execBG(codeObj, localVars = defLocals, globalVars = defGlobals):
    '''Executes a code object in a background thread, using the given namespace.
    
    Returns
    -------
    t : thread
        The thread in which the code is executing (can be used with threading.join later)
    '''
    t = threading.Thread(target=_exec, args = (codeObj, localVars, globalVars))
    t.start()
    return t

def execFile(filename, localVars = defLocals, globalVars = defGlobals):
    #fid = open(checkFilename(filename))
    #code = fid.read()
    #fid.close()

    _execfile(checkFilename(filename), localVars, globalVars)

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
    except Exception as e:
        _exec("splash.SetMessage('%s', 'Initialising %s ... FAIL')" % (name,name), defGlobals, defLocals)
        raise e


def InitBG(name, codeObj):
    '''Runs a portion of the initialisation code in a background thread
    
    Parameters
    ----------
    name : string
        A descriptive name for the code block - e.g. 'Camera'
    codeObj : string, compiled code object
        The code that will be executed - something that `exec` understands
    
    Returns
    -------
    t : thread
        The thread in which the code is executing (can be used with threading.join later)    
    '''
    t = threading.Thread(target=_bginit, args = (name,codeObj))
    t.start()
    bgInitThreads.append(t)
    return t
    

def joinBGInit():
    '''
    Wait for all the initialisation tasks that bave been launched as background 
    threads to complete.
    '''
    for t in bgInitThreads:
        print(t)
        t.join()

def InitGUI(code):
    '''Add a piece of code to a list of items to be executed once the GUI is 
    up and running. Used to defer the initialisation of GUI components ascociated
    with hardware items until they can be displayed.
    
    Parameters
    ----------
    codeObj : string, compiled code object
        The code that will be executed - something that `exec` understands  
    '''
    defLocals['postInit'].append(code)
