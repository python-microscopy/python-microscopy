import warnings
import logging
import inspect

logger = logging.getLogger(__name__)

def _have_gui():
    '''Test to see if we have a GUI available'''
    try:
        import wx
        if wx.GetApp() is not None:
            return True
    except ImportError:
        pass
            
    return False

class NoninteractiveContext(object):
    '''Context manager to indicate that we are running in a non-interactive 
    environment - e.g. recipe execution. This will suppress wx.MessageBox
    warnings even if a GUI is available.
    
    The code is used through the `noninteractive` singleton instance.
    
    Example:

    >>> with PYME.warnings.noninteractive:
    >>>     # do something that might trigger a warning
    >>>     # warning will be logged, but no GUI popup will be shown

    '''
    
    def __init__(self):
        self._entrycount = 0

    def __enter__(self):
        self._entrycount += 1
    
    def __exit__(self, *args):
        self._entrycount -= 1

    def __bool__(self):
        return self._entrycount > 0
    
noninteractive = NoninteractiveContext()

def _gui_warning(message, allow_cancel=False):
    '''Show a GUI warning dialog (MessageBox) if
    we are running in a interactive context and a GUI is available.

    Returns True if the user clicked OK, False if they clicked Cancel (and Cancel was offered as an option)
    
    '''
    if _have_gui() and not noninteractive:
        import wx
        return wx.MessageBox(message, 'Warning', wx.OK |wx.ICON_EXCLAMATION | (wx.CANCEL if allow_cancel else 0)) == wx.OK
    else:
        return True

def warn(message, category=None, stacklevel=1, allow_cancel=False):
    warnings.warn(message, category, stacklevel=stacklevel+1)
    
    #where was the call to PYME.warnings.warn made from?
    calling_frame = inspect.stack()[1]
    calling_module = inspect.getmodule(calling_frame[0])
    calling_lineno = calling_frame[2]

    logger.warning('Warning from %s:%d: %s' % (calling_module.__name__, calling_lineno, message))
    
    return _gui_warning(message, allow_cancel=allow_cancel)