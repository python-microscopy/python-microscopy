import wx

class FlipMirrorControl(wx.Panel):
    def __init__(self, flipper, parent, winid=-1):
        """
        Simple panel to facilitate changing a motorized flip mirror position
        through the GUI
        flipper: PYME.Acquire.Hardware.thorlabs_mff_flipper.ThorlabsMFF
        """
        self.flipper = flipper
        wx.Panel.__init__(self, parent, winid)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Position:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        self.pos_select = wx.ComboBox(self, -1, choices=self.flipper.position_names,
                                      value= '%d' % self.flipper.GetPos(), 
                                      size=(65, -1), style=wx.CB_READONLY|wx.TE_PROCESS_ENTER)
        self.pos_select.Bind(wx.EVT_COMBOBOX, self.OnComboBox)
        hsizer.Add(self.pos_select, 0, wx.ALL, 2)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        self.SetSizerAndFit(vsizer)


    def OnComboBox(self, wx_event=None):
        cb = wx_event.GetEventObject()
        self.flipper.SetPosByName(cb.GetValue())
    
    def update(self):
        self.pos_select.SetValue(self.flipper.GetPosName())


# example init script addition:
# @init_hardware('filter flipper')
# def filter_flipper(scope):
#     from PYME.Acquire.Hardware import thorlabs_mff_flipper
#     scope.ff = thorlabs_mff_flipper.ThorlabsMFF('37004323', name='PumpQWP', position_names=['Blank', 'QWP'])
#     # POS 1 = nothing
#     # POS 2 = QWP in path
#     scope.ff.register(scope.state)
#
# @init_gui('filter flip panel')
# def filter_flip_panel(MainFrame, scope):
#     from PYME.Acquire.ui import flipper_panel
#     fm = flipper_panel.FlipMirrorControl(scope.ff, MainFrame.toolPanel)
#     MainFrame.time1.WantNotification.append(fm.update)
#     MainFrame.camPanels.append((fm, 'Pump QWP Flipper'))
