import wx

class DelayPanel(wx.Panel):
    """
    Simple slider panel showing/controling the delay of the MPD picosecond 
    delay module. See PYME.Acquire.Hardware.mpd_picosecond_delayer.

    Example setup in init script:
    @init_gui('pulse phase control')
    def pulse_phase_controls(MainFrame, scope):
        from PYME.Acquire.ui import mpd_picosecond_delay_panel
        delay_panel = mpd_picosecond_delay_panel.DelayPanel(MainFrame, scope.mpd, scope)
        MainFrame.camPanels.append((delay_panel, 'Phase Delay', False, False))
        MainFrame.time1.WantNotification.append(delay_panel.update)
    """
    def __init__(self, parent, delayer, scope):
        self.delayer = delayer
        self.scope = scope
        self.sliding = False

        wx.Panel.__init__(self, parent)
        vsizer=wx.BoxSizer(wx.VERTICAL)

        delay = wx.StaticBoxSizer(wx.StaticBox(self, -1, u'Delay (ps)'), wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.l = wx.StaticText(self, -1, '%d' % self.delayer.delay)
        hsizer.Add(self.l, 0, wx.ALL, 2)
        self.sl = wx.Slider(self, -1, self.delayer.delay, 0, self.delayer.GetMaxDelay(), size=wx.Size(150,-1),style=wx.SL_HORIZONTAL | wx.SL_HORIZONTAL | wx.SL_AUTOTICKS )
        self.sl.SetTickFreq(25000)
        self.Bind(wx.EVT_SCROLL,self.on_slide)
        hsizer.Add(self.sl, 1, wx.ALL|wx.EXPAND, 2)
        delay.Add(hsizer, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        

        vsizer.Add(delay, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cb_highspeed = wx.CheckBox(self, wx.ID_ANY, 'High-Speed mode (no GUI update)')
        self.cb_highspeed.SetValue(self.delayer.high_speed_mode)
        self.cb_highspeed.Bind(wx.EVT_CHECKBOX, self.on_cb_highspeed)

        vsizer.Add(hsizer, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.SetSizer(vsizer)

        

    def on_slide(self, event):
        self.sliding = True
        try: 
            sl = event.GetEventObject()
            self.delayer.delay = sl.GetValue()
            self.l.SetLabel('%d' % self.delayer.delay)
        finally:
            self.sliding = False
    
    def on_cb_highspeed(self, wx_event):
        self.delayer.high_speed_mode = self.cb_highspeed.GetValue()

    def update(self):
        # only update if we aren't sliding and if we aren't running
        # the delayer in high speed mode (where it doesn't even update
        # the front panel for the sake of speed)
        if (not self.sliding) and (not self.delayer.high_speed_mode):
            delay = self.delayer.GetDelay()
            self.sl.SetValue(delay)
            self.l.SetLabel('%d' % delay)
