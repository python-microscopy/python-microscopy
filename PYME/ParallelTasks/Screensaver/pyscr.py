"""Win32 Screensaver class, This class imlements the event loop and handling.
It terminates the screensaver on appropriate events.

To write a screensaver, derrive a class from pyscr.Screensaver and implement
the following methods:
    - initialized   called once on init
    - finalize      called once on exit
    - tick          called periodicaly, set with TIMEBASE
    - configure     called when the screensaver has to be configured

The TIMEBASE class attribute determines the interval for the tick() calls.
The value is a time in seconds. If set to None, it will never call the
tick() method.

The user module has to contain exactly one class derrived from
pysrc.Screensaver. It is automaticaly searched. The user module just has
to call pyscr.main() - that's all!

(C) 2003 Chris Liechti <cliechti@gmx.net>
This is distributed under a free software license, see license.txt.
"""

import sys, traceback
from io import StringIO
from ctypes import *
from winuser import*
from wingdi import *
from wincommon import *
import windc

VerifyScreenSavePwd = None

SCRM_VERIFYPW = WM_APP  #alias for user message


class Screensaver:
    """Win32 Screensaver base class.
    User has to override all or some of 'configure', 'initialize', 'finalize'
    and 'tick' and set TIMEBASE if tick() is to be used.
    
    Attributes:
    hWnd:
        This is the window handle of the drawing surface. This handle can be
        passed to other libs such as win32all.
        
    dc:
        This is the wrapped drawing surface using windc.
    
    width and height:
        These attributes contain the size of the drawing area.

    """
    
    #------------------------------------------------------------------------
    # Attributes to be changeable by the user:
    #------------------------------------------------------------------------
    TIMEBASE = None     # tick() interval in secods
    
    #------------------------------------------------------------------------
    # Methods that are overriden by the user:
    #------------------------------------------------------------------------
    def configure(self):
        """Called to open the screensaver configuration.
           Maybe implement here something with venster or win32all"""
    
    def initialize(self):
        """Called once when the screensaver is started."""

    def finalize(self):
        """Called when the screensaver terminates."""
    
    def tick(self):
        """Called when the timer tick occours, set up with TIMEBASE."""
    
    #------------------------------------------------------------------------
    # Internal methods
    #------------------------------------------------------------------------
    def __init__(self):
        self.closing = False
        self.fChildPreview = False
        self.pt_orig = POINT()
        self.checking_pwd = False
        self.width = 0
        self.height = 0
        self.hWnd = None
        self._finalization_done = False

    
    def _screenSaverEventHandler(self, hWnd, msg, wParam, lParam):
        """Event handler that looks for events and terminates the screensaver."""
        self.hWnd = hWnd
        if msg == WM_CREATE:            #called on startup
            self.dc = windc.DC(self.hWnd)
            #now start a timer if the user wants to
            try:
                self.initialize()
            except:
                PostMessage(self.hWnd, WM_CLOSE, 0, 0)
                exinfo = StringIO()
                traceback.print_exc(file=exinfo)
                print(("-"*78))
                print("Exception in users initialiaztion code:")
                print((exinfo.getvalue()))
                print(("-"*78))
                self.closing = True
                self.finalize()
                windll.user32.MessageBoxA(self.hWnd, exinfo.getvalue(), "Error in pyscr startup", 0, 0, 0)
                return True
                #~ return False
            if self.TIMEBASE is not None:
                windll.user32.SetTimer(self.hWnd, 100, int(self.TIMEBASE*1000), NULL)
        elif msg == WM_TIMER:           #timer ticks...
            if not self.closing:
                self.tick()
            #~ return True
        #~ elif msg == WM_DESTROY:         #called on close
            #~ self.finalize()
            
        
        #don't do any special processing when in preview mode
        if self.fChildPreview and msg == WM_CLOSE:
            if not self._finalization_done:
                self._finalization_done = True
                self.finalize()
        
        if self.fChildPreview or self.closing:
            return DefWindowProc(hWnd, msg, wParam, lParam)
    
        #other messages following...
        if msg == WM_CLOSE:
            #TerminateScreenSaver:
            #don't allow recursion
            if self.checking_pwd or self.closing:
                return
            #verify password
            if VerifyScreenSavePwd:
                self.checking_pwd = True
                self.closing = SendMessage(hWnd, SCRM_VERIFYPW, 0, 0)
                self.checking_pwd = False
            else:
                self.closing = True
        
            #are we closing?
            if self.closing:
                if not self._finalization_done:
                    self._finalization_done = True
                    self.finalize()
                DestroyWindow(hWnd)
            else:
                GetCursorPos(byref(self.pt_orig)) #if not: get new mouse position
    
            #do NOT pass this to DefWindowProc; it will terminate even if an invalid password was given.
            return False
        elif msg == SCRM_VERIFYPW:
            #verify password or return TRUE if password checking is turned off
            if VerifyScreenSavePwd:
                return VerifyScreenSavePwd(hWnd)
            else:
                return True
        elif msg == WM_SETCURSOR:
            if not self.checking_pwd:
                SetCursor(NULL)
                return True
        #~ elif msg in (WM_NCACTIVATE, WM_ACTIVATE, WM_ACTIVATEAPP):
            #~ if wParam != False:
                #~ break;
        elif msg == WM_MOUSEMOVE:
            pt = POINT()
            GetCursorPos(byref(pt))
            #check if mouse has moved, give it some threshold
            if abs(pt.x - self.pt_orig.x) > 10 or abs(pt.y != self.pt_orig.y) > 10:
                if not self.checking_pwd:
                    PostMessage(hWnd, WM_CLOSE, 0, 0)
        elif msg in (WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN, WM_KEYDOWN, WM_SYSKEYDOWN):
            #try to terminate screen saver
            if not self.checking_pwd:
                PostMessage(hWnd, WM_CLOSE, 0, 0)
        return DefWindowProc(hWnd, msg, wParam, lParam)
    
    def _topLevelEventHandler(self, hWnd, msg, wParam, lParam):
        """Event handler that is registered at the window."""
        try:
            if msg == WM_CREATE:
                if not self.fChildPreview:
                    SetCursor(NULL)     #hide mouse
                    #mouse is not supposed to move from this position
                    GetCursorPos(byref(self.pt_orig))
            elif msg == WM_DESTROY:
                PostQuitMessage(0)
            elif msg == WM_PAINT:
                if self.closing:
                    return DefWindowProc(hWnd, msg, wParam, lParam)
            elif msg == WM_SYSCOMMAND:
                if not self.fChildPreview:
                    if wParam in (SC_CLOSE, SC_SCREENSAVE, SC_NEXTWINDOW, SC_PREVWINDOW):
                        return False
            elif msg in (WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_RBUTTONDOWN,
                         WM_MBUTTONDOWN, WM_KEYDOWN, WM_SYSKEYDOWN,
                         WM_NCACTIVATE, WM_ACTIVATE, WM_ACTIVATEAPP,
            ):
                if self.closing:
                    return DefWindowProc(hWnd, msg, wParam, lParam)
            return self._screenSaverEventHandler(hWnd, msg, wParam, lParam)
        except:
            PostMessage(hWnd, WM_CLOSE, 0, 0)
            raise
    
    
    def launchScreenSaver(self, hParent=None):
        """Start screensaver"""
        hMainInstance = GetModuleHandle(NULL)
        CLASS_SCRNSAVE = "WindowsScreenSaverClass"
        
        cls = WNDCLASS()
        
        cls.hCursor = NULL
        #~ cls.hIcon = LoadIcon(hMainInstance, MAKEINTATOM(ID_APP))
        cls.lpszMenuName = NULL
        cls.lpszClassName = CLASS_SCRNSAVE
        cls.hbrBackground = GetStockObject(BLACK_BRUSH)
        cls.hInstance = hMainInstance
        cls.style = CS_VREDRAW | CS_HREDRAW | CS_SAVEBITS | CS_PARENTDC
        cls.lpfnWndProc = WNDPROC(self._topLevelEventHandler)
        cls.cbWndExtra = 0
        cls.cbClsExtra = 0
        #~ print cls
        
        RegisterClass(byref(cls))
    
        rc = RECT()
        #a slightly different approach needs to be used when displaying in a preview window
        if hParent:
            style = WS_CHILD
            GetClientRect(hParent, byref(rc))
        else:
            style = WS_POPUP
            rc.right = GetSystemMetrics(SM_CXSCREEN)
            rc.bottom = GetSystemMetrics(SM_CYSCREEN)
            #~ #XXX DEBUG
            #~ rc.right = 200
            #~ rc.bottom = 200
            style |= WS_VISIBLE
            
        self.width = rc.right
        self.height = rc.bottom
        
        #~ print "size: ", rc, not hParent and WS_EX_TOPMOST or 0
        
        #create main screen saver window
        self.hMainWindow = CreateWindowEx(not hParent and WS_EX_TOPMOST or 0,
                                    CLASS_SCRNSAVE,
                                    "SCREENSAVER",
                                    style,
                                    0, 0, rc.right, rc.bottom,
                                    hParent or NULL,
                                    NULL,
                                    hMainInstance,
                                    NULL)
    
        #display window and start pumping messages
        msg = MSG()
        if self.hMainWindow:
            UpdateWindow(self.hMainWindow);
            ShowWindow(self.hMainWindow, SW_SHOW);
    
            while GetMessage(byref(msg), 0, 0, 0):
                TranslateMessage(byref(msg))
                DispatchMessage(byref(msg))
                #~ print msg
        return msg.wParam
    
    def launchConfig(self):
        """command to start the sceensaver configuration dialog"""
        self.configure()
    
    def screenSaverChangePassword(self, hParent):
        #this does not seem to work on WinXP
        
        #load Master Password Router (MPR)
        windll.mpr.PwdChangePasswordA("SCRSAVE", hParent, 0, NULL)

def main():
    """Call this function in the user module. It will automaticaly scan for a
    class derrived from Screensaver and launch it. If more than one such class
    is there it's random which one is used... so only define one per file."""
    
    import __main__
    for name, value in __main__.__dict__.items():
        if type(value) == type(Screensaver) and issubclass(value, Screensaver):
            saver = value()
            break
    else:
        raise Exception("No Screensaver class found")

    for i in range(1, len(sys.argv)):
        opt = sys.argv[i].lower()
        if opt.startswith("/"):
            opt = opt[1:]
        
        if opt == 's':
            #start screen saver
            return saver.launchScreenSaver()
        elif opt == 'p':
            #start screen saver in preview window
            hParent = int(sys.argv[i+1])
            if hParent and IsWindow(hParent):
                saver.fChildPreview = True
                return saver.launchScreenSaver(hParent)
        elif opt == 'c':
            #display configure dialog
            saver.launchConfig()
            return 0
        elif opt == 'a':
            #change screen saver password
            hParent = int(sys.argv[i+1])
            if not hParent or not IsWindow(hParent):
                hParent = GetForegroundWindow()
            saver.screenSaverChangePassword(hParent)
            return 0
    
    saver.launchConfig()
    return 0

if __name__ == '__main__':
    #test
    sys.exit(main())