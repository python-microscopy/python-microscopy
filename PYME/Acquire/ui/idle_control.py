#!/usr/bin/python
# -*- coding: utf-8 -*-
#  Idle mode control panel for PYMEAcquire

import wx
import logging

logger = logging.getLogger(__name__)

class IdleModeControl(wx.Panel):
    """
    Two-button (Play/Stop) control for setting microscope idle/active state.

    - ▶ Play: exit Idle (set active).
    - ⏹ Stop: enter Idle. 
    
    """
    def __init__(self, parent, scope):
        wx.Panel.__init__(self, parent)
        self.scope = scope
        self.parent = parent

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        # Label
        hsizer.Add(wx.StaticText(self, -1, "Microscope:"), 0,
                   wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        
        green = wx.Colour(0, 170, 0)
        red = wx.Colour(200, 0, 0)

        # Play (exit idle)
        self.btnPlay = wx.Button(self, -1, "▶", style=wx.BU_EXACTFIT)
        self.btnPlay.SetToolTip(wx.ToolTip("Set Active (exit Idle)"))
        self.btnPlay.Bind(wx.EVT_BUTTON, self.on_play)
        self.btnPlay.SetForegroundColour(green)
        hsizer.Add(self.btnPlay, 0, wx.ALL, 2)

        # Stop (enter idle)
        self.btnStop = wx.Button(self, -1, "⏹", style=wx.BU_EXACTFIT)
        self.btnStop.SetToolTip(wx.ToolTip("Set Idle (pause camera)"))
        self.btnStop.Bind(wx.EVT_BUTTON, self.on_stop)
        self.btnStop.SetForegroundColour(red)
        hsizer.Add(self.btnStop, 0, wx.ALL, 2)

        self.SetSizerAndFit(hsizer)

        # Listen for state changes from the frameWrangler
        self.scope.frameWrangler.onIdleChange.connect(self.on_idle_changed)
        self.scope.frameWrangler.onStart.connect(lambda *_args, **_kw: wx.CallAfter(self.update))
        self.scope.frameWrangler.onStop.connect(lambda *_args, **_kw: wx.CallAfter(self.update))

        # Initial state
        self.update()

    def on_play(self, event=None):
        """Exit idle mode (set Active)."""
        try:
            if not self.scope.frameWrangler.get_idle():
                return  # already active -> no-op
            logger.info('User setting microscope to ACTIVE')
            self.scope.frameWrangler.set_idle(False)
        except Exception:
            logger.exception('Error setting Active state from IdleModeControl')
        finally:
            wx.CallAfter(self.update)

    def on_stop(self, event=None):
        """Enter idle mode (pause camera)."""
        try:
            if self.scope.frameWrangler.get_idle():
                return  # already idle -> no-op
            logger.info('User setting microscope to IDLE')
            self.scope.frameWrangler.set_idle(True)
        except Exception:
            logger.exception('Error setting Idle state from IdleModeControl')
        finally:
            wx.CallAfter(self.update)

    def on_idle_changed(self, sender, idle, **kwargs):
        """Called when the idle state changes (from signal)"""
        wx.CallAfter(self.update)

    def update(self):
        """enable/disable buttons based on idle/active."""
        idle = self.scope.frameWrangler.get_idle()

        if idle:  # Idle state
            self.btnPlay.SetToolTip(wx.ToolTip("Set Active (exit Idle)"))
            self.btnStop.SetToolTip(wx.ToolTip("Microscope is idle"))
            self.btnPlay.Enable()
            self.btnStop.Disable()
        else:  # Active state
            self.btnPlay.SetToolTip(wx.ToolTip("Microscope is active"))
            self.btnStop.SetToolTip(wx.ToolTip("Set Idle (pause camera)"))
            self.btnPlay.Disable()
            self.btnStop.Enable()

        self.Layout()
