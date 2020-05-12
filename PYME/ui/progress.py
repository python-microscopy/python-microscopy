"""
Defines a context manager for long running processes. It does the following:

- changes the cursor to a timer
- displays text to indicate what is running in the status bar (if window has one)
- displays a dialog with a traceback if an error occurs
"""

import wx
import traceback

_package_info = '''python-microscopy={pyme_version}
python={python_version}, platform={platform}
numpy={numpy_version}, wx={wx_version}'''

def get_package_versions():
    import sys
    import PYME.version
    import numpy as np
    
    # TODO - get other package versions?
    
    return _package_info.format(pyme_version=PYME.version.version,
                                python_version='.'.join([str(n) for n in sys.version_info[:3]]), platform=sys.platform,
                                numpy_version=np.version.version, wx_version=wx.version())

error_msg = ''' Error whilst {description}
==============================================
{pkgversions}

Traceback
=========
{traceback}
'''

class TracebackDialog(wx.Dialog):
    def __init__(self, parent, description, exc_type, exc_val, exc_tb):
        wx.Dialog.__init__(self, parent, title='Error whilst %s' % description)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        vsizer.Add(wx.StaticText(self, label='%s(%s)' %(exc_type.__name__, exc_val)),0, wx.ALL, 2 )
        #print(exc_tb)
        tb = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
        #print(tb, type(tb))
        vsizer.Add(wx.TextCtrl(self, value=error_msg.format(description=description, pkgversions=get_package_versions(), traceback=tb), size=(400,300), style=wx.TE_MULTILINE|wx.TE_READONLY), 0, wx.ALL, 2)
        vsizer.Add(wx.StaticText(self, label='Copy and paste the above message into an email or issue when\n reporting a bug'), 0, wx.ALL, 2)
        
        vsizer.Add(self.CreateButtonSizer(wx.OK), 0, wx.TOP, 2)
        self.SetSizerAndFit(vsizer)
        
def show_traceback_dialog(parent, description, exc_type, exc_val, exc_tb):
    dlg = TracebackDialog(parent, description, exc_type, exc_val, exc_tb)
    dlg.ShowModal()
    dlg.Destroy()

class ComputationInProgress(object):
    """
    Context manager for wrapping long-running tasks in the GUI - indicates that something is happening and display a
    dialog showing the error on failure
    """
    def __init__(self, window, short_description):
        self.window = window # type: wx.Window
        self.description = short_description
        
    def __enter__(self):
        self._old_cursor = self.window.GetCursor()
        self.window.SetCursor(wx.StockCursor(wx.CURSOR_WAIT))
        
        if hasattr(self.window, 'SetStatus'):
            # at present, only VisGUI has SetStatus - possibly rename...
            self.window.SetStatus('%s ... in progress' % self.description)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.window.SetCursor(self._old_cursor)
        
        if exc_type:
            show_traceback_dialog(self.window, self.description, exc_type, exc_val, exc_tb)

            if hasattr(self.window, 'SetStatus'):
                self.window.SetStatus('%s ... [FAILED]' % self.description)
                
        else:
            if hasattr(self.window, 'SetStatus'):
                self.window.SetStatus('%s ... [COMPLETE]' % self.description)
        