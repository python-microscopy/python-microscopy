"""

Workaround for a fortran module linked from scipy.stats which causes a hard exit if a process is terminated using cntrl-c

See: http://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats

for details.

"""

import os
import sys

if sys.platform == 'win32':
    try:
        import imp
        import ctypes
        import thread
        import win32api
        
        # Load the DLL manually to ensure its handler gets
        # set before our handler.
        #basepath = imp.find_module('numpy')[1]
        #ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
        #ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

        #anaconda stores dlls in a Library/bin directory
        basepath = os.path.join(sys.prefix, 'Library', 'bin')
        ctypes.CDLL(os.path.join(basepath, 'libmmd.dll'))
        ctypes.CDLL(os.path.join(basepath, 'libifcoremd.dll'))
        
        # Now set our handler for CTRL_C_EVENT. Other control event
        # types will chain to the next handler.
        def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
            if dwCtrlType == 0: # CTRL_C_EVENT
                hook_sigint()
                return 1 # don't chain to the next handler
            return 0 # chain to the next handler
        
        win32api.SetConsoleCtrlHandler(handler, 1)
    except:
        import logging
        logging.exception('Could not locate numpy fortran libs, ignoring')