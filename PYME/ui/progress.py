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

error_msg = ''' Error whilst running {description}
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
        
        hsizer=wx.BoxSizer(wx.HORIZONTAL)
        s_b = wx.StaticBitmap(self)
        s_b.SetIcon(wx.ArtProvider.GetIcon(wx.ART_ERROR, client=wx.ART_MESSAGE_BOX, size=(32,32)))
        hsizer.Add(s_b, 0, wx.ALL|wx.ALIGN_CENTRE_VERTICAL, 2)
        hsizer.Add(wx.StaticText(self, label='%s(%s)' %(exc_type.__name__, exc_val)),0, wx.ALL|wx.ALIGN_CENTRE_VERTICAL, 2 )
        vsizer.Add(hsizer, 0, wx.ALL, 2)
        #print(exc_tb)
        tb = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
        #print(tb, type(tb))
        self.err_text=wx.TextCtrl(self, value=error_msg.format(description=description, pkgversions=get_package_versions(), traceback=tb),
                               size=(500,300), style=wx.TE_MULTILINE|wx.TE_READONLY)
        vsizer.Add(self.err_text, 0, wx.ALL|wx.EXPAND, 15)
        vsizer.Add(wx.StaticText(self, label='Copy and paste the above message into an email or issue when\n reporting a bug'),0, wx.ALL, 5)

        btnsizer = wx.StdDialogButtonSizer()
        
        copy_btn = wx.Button(self, label="Copy")
        copy_btn.Bind(wx.EVT_BUTTON, self.on_copy)
        btnsizer.Add(copy_btn, 0, wx.LEFT|wx.ALIGN_CENTER_VERTICAL,5)
        
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)
        
        btnsizer.Realize()
                
        vsizer.Add(btnsizer, 0, wx.ALL|wx.EXPAND, 2)
        self.SetSizerAndFit(vsizer)
    
    def on_copy(self, event):
        self.data = wx.TextDataObject()
        self.data.SetText(self.err_text.GetValue())
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(self.data)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Unable to open the clipboard","Error")
        
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
        self.window.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        
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
 
 
 
def managed(fcn, window, description):
    """ wrap a function in a context manager - used to wrap menu items"""
    def func(*args, **kwargs):
        with ComputationInProgress(window, description):
            return fcn(*args, **kwargs)
            
    return func


class SlowComputation(object):
    """ Pops up a dialog with a launches a computation in a separate thread """
    