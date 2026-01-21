
import wx

class ManualEventPanel(wx.Panel):
    def __init__(self, parent):
        """
        Panel to allow manually logging an acquisition event in PYMEAcquire.

        Parameters
        ----------
        parent : wx parent, i.e. MainFrame in your init script

        example usage:
            @init_gui('events')
            def events(MainFrame, scope):
                from PYME.Acquire.ui.manual_event_panel import ManualEventPanel
                event_panel = ManualEventPanel(MainFrame)
                MainFrame.camPanels.append((event_panel, 'manual event panel', False))
        """
        wx.Panel.__init__(self, parent)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Name:'), 0, wx.ALL, 2)
        self.name = wx.TextCtrl(self, -1, value='')
        hsizer.Add(self.name, 1, wx.EXPAND)
        vsizer.Add(hsizer, 1, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Description:'), 0, wx.ALL, 2)
        self.description = wx.TextCtrl(self, -1, value='')
        hsizer.Add(self.description, 1, wx.EXPAND)
        vsizer.Add(hsizer, 1, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.log_button = wx.Button(self, -1, 'Log Event')
        hsizer.Add(self.log_button, 0, wx.ALL, 2)
        self.log_button.Bind(wx.EVT_BUTTON, self.OnLog)
        vsizer.Add(hsizer, 1, wx.EXPAND)

        self.SetSizerAndFit(vsizer)

    def OnLog(self, wx_event=None):
        """    
        """
        from PYME.Acquire import eventLog
        eventLog.logEvent(self.name.GetValue(), self.description.GetValue())
