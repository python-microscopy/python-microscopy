"""Some win32 stuff that is used in the pyscr and windc module.

(C) 2003 Chris Liechti <cliechti@gmx.net>
This is distributed under a free software license, see license.txt.
"""
#some parts that are used in different files in the screensaver modules

from ctypes import *
from winuser import*
from wingdi import *

HANDLE = c_ulong
HWND   = c_int
HBRUSH = HCURSOR = HICON = HANDLE
LPCSTR = c_char_p
LPARAM = c_long
DWORD = WPARAM = c_int
NULL = 0
LONG = c_long
BYTE = c_byte
TCHAR = c_char

def ValidHandle(value):
    if value == 0:
        raise WinError()
    return value

WNDPROC = WINFUNCTYPE(c_int, HWND, c_uint, WPARAM, LPARAM)

class PrettyPrint:
    """ctypes scruct should reveal their information when printed.
       easy for debuging purposes."""
    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            ', '.join(['%s=%r' % (field, getattr(self, field))
                       for field, type in self._fields_
                      ])
        )
            
class RECT(PrettyPrint, Structure):
    _fields_ = [('left',        c_int),
                ('right',       c_int),
                ('top',         c_int),
                ('bottom',      c_int)]

class WNDCLASS(PrettyPrint, Structure):
    _fields_ = [('style',       c_uint),
                ('lpfnWndProc', WNDPROC),
                ('cbClsExtra',  c_int),
                ('cbWndExtra',  c_int),
                ('hInstance',   HANDLE),
                ('hIcon',       HANDLE),
                ('hCursor',     HCURSOR),
                ('hbrBackground', HBRUSH),
                ('lpszMenuName',  c_void_p), #LPCSTR),
                ('lpszClassName', LPCSTR),
    ]

class POINT(PrettyPrint, Structure):
    _fields_ = [('x', c_long), ('y', c_long)]

class RECT(PrettyPrint, Structure):
    _fields_ = [('left', c_long),
                ('top', c_long),
                ('right', c_long),
                ('bottom', c_long),
               ]

class MSG(PrettyPrint,  Structure):
    _fields_ = [('hwnd',        HWND),
                ('message',     c_uint),
                ('wParam',      WPARAM),
                ('lParam',      LPARAM),
                ('time',        DWORD),
                ('pt',          POINT),
    ]


LF_FACESIZE = 32

class LOGFONT(Structure):
    _fields_ = [("lfHeight", LONG),
                ("lfWidth", LONG),                
                ("lfEscapement", LONG),
                ("lfOrientation", LONG),
                ("lfWeight", LONG),
                ("lfItalic", BYTE),
                ("lfUnderline", BYTE),
                ("lfStrikeOut", BYTE),
                ("lfCharSet", BYTE),
                ("lfOutPrecision", BYTE),
                ("lfClipPrecision", BYTE),
                ("lfQuality", BYTE), 
                ("lfPitchAndFamily", BYTE),
                ("lfFaceName", TCHAR * LF_FACESIZE)]


GetModuleHandle = windll.kernel32.GetModuleHandleA
GetModuleHandle.restype = ValidHandle

CreateWindowEx = windll.user32.CreateWindowExA
CreateWindowEx.argtypes = [
    #[DWORD,  LPCWSTR,  LPCWSTR,  DWORD,   int,   int,   int,   int,   HWND,  HMENU,  HINSTANCE, LPVOID]
     c_ulong, c_char_p, c_char_p, c_ulong, c_int, c_int, c_int, c_int, HANDLE,HANDLE, HANDLE,    c_ulong
]
CreateWindowEx.restype = ValidHandle

RegisterClass = windll.user32.RegisterClassA
RegisterClass.restype = ValidHandle

UpdateWindow = windll.user32.UpdateWindow
ShowWindow = windll.user32.ShowWindow
DestroyWindow = windll.user32.DestroyWindow
GetMessage = windll.user32.GetMessageA
TranslateMessage = windll.user32.TranslateMessage
DispatchMessage = windll.user32.DispatchMessageA
PostMessage = windll.user32.PostMessageA
PostQuitMessage = windll.user32.PostQuitMessage
GetClientRect= windll.user32.GetClientRect
GetSystemMetrics = windll.user32.GetSystemMetrics
GetStockObject = windll.gdi32.GetStockObject
DefWindowProc = windll.user32.DefWindowProcA
SetCursor = windll.user32.SetCursor
GetCursorPos = windll.user32.GetCursorPos
IsWindow = windll.user32.IsWindow
GetForegroundWindow = windll.user32.GetForegroundWindow

GetDC = windll.user32.GetDC
