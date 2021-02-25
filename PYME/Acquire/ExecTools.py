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
"""Defines a few neat things to allow scripts to be executed in background during
initialisation & to allow a user script directory"""
import threading
#import pythoncom

import os
import sys

import logging
logger = logging.getLogger(__name__)

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

#bgInitThreads = []
bg_init_tasks = []
#bgInitStatus = {}

class HWNotPresent(Exception):
    """An exception which, if thrown during initialisation, will result in a more
    informative error message than just failure"""
    pass


def setDefaultNamespace(locals, globals):
    """Populate the namespace in which the initialisation script will run.
    
    Parameters
    ----------
    locals : dict
    globals : dict    
    """
    global defGlobals
    global defLocals

    defLocals = locals
    defGlobals = globals

from PYME.util.execfile import _exec, _execfile

def execBG(codeObj, localVars = defLocals, globalVars = defGlobals):
    """Executes a code object in a background thread, using the given namespace.
    
    Returns
    -------
    t : thread
        The thread in which the code is executing (can be used with threading.join later)
    """
    t = threading.Thread(target=_exec, args = (codeObj, localVars, globalVars))
    t.start()
    return t

def execFile(filename, localVars = defLocals, globalVars = defGlobals, then=None):
    #fid = open(checkFilename(filename))
    #code = fid.read()
    #fid.close()

    _execfile(filename, localVars, globalVars)
    if then is not None:
        then()

def execFileBG(filename, localVars = defLocals, globalVars = defGlobals, then=None):
    t = threading.Thread(target=execFile, args = (filename, localVars, globalVars, then))
    t.start()
    #return the thread so we can join it ...
    return t

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

class BGInitTask(object):
    TASK_RUNNING, TASK_DONE, TASK_FAILED, NOT_PRESENT = range(4)
    _status_codes = ['', 'DONE', 'FAIL', 'NOT PRESENT']
    
    def __init__(self, name, codeObj):
        self.name = name
        self.status = self.TASK_RUNNING
        self.codeObj = codeObj
        
        self.thread = threading.Thread(target=self._bginit)
        self.thread.start()

    def _bginit(self):
        global defGlobals
        global defLocals
        self.status = self.TASK_RUNNING
        try:
            if callable(self.codeObj):
                self.codeObj(defLocals['scope'])
            else:
                _exec(self.codeObj, defGlobals, defLocals)
            self.status = self.TASK_DONE
        except HWNotPresent:
            self.status = self.NOT_PRESENT
        except Exception as e:
            self.status = self.TASK_FAILED
            logger.exception('Error running background init task %s' % self.name)
            raise e
        
    def get_status_msg(self):
        return 'Initialising %s ... %s' % (self.name, self._status_codes[self.status])
    
    def join(self, timeout=None):
        self.thread.join(timeout)
        

def InitBG(name, codeObj):
    """Runs a portion of the initialisation code in a background thread
    
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
    """
    t = BGInitTask(name,codeObj)
    bg_init_tasks.append(t)
    return t
    

def joinBGInit():
    """
    Wait for all the initialisation tasks that bave been launched as background 
    threads to complete.
    """
    for t in bg_init_tasks:
        #print(t)
        t.join()

class GUIInitTask(object):
    def __init__(self, name, codeObj):
        self.name = name
        self.codeObj = codeObj

    def run(self, parent, scope):
        """

        Parameters
        ----------
        parent : PYME.Acquire.acquiremainframe.PYMEMainFrame, wx.Frame
        scope : PYME.Acquire.microscope.microscope
        
        """
        global defGlobals
        global defLocals
        #try:
        if callable(self.codeObj):
            self.codeObj(parent, scope)
        else:
            _exec(self.codeObj, defGlobals, defLocals)

        #except Exception as e:
        #    raise e

def InitGUI(code='', name=''):
    """Add a piece of code to a list of items to be executed once the GUI is
    up and running. Used to defer the initialisation of GUI components ascociated
    with hardware items until they can be displayed.
    
    Parameters
    ----------
    codeObj : string, compiled code object
        The code that will be executed - something that `exec` understands  
    """
    defLocals['postInit'].append(GUIInitTask(name, code))

#define decorators
def init_gui(name):
    def _init_gui(fcn):
        return InitGUI(fcn, name)
        
    return _init_gui
        
def init_hardware(name):
    def _init_hw(fcn):
        return InitBG(name, fcn)
    
    return _init_hw