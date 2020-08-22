# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 16:29:38 2015

@author: David Baddeley
"""

import wx

class EnumControl(wx.Panel):
    def __init__(self, parent, target, targetPropertyName=None):
        wx.Panel.__init__(self, parent)
        self.scope = parent.scope
        self.parent = parent
        self.target = target
        # this keyword allows us to override unwieldy property Names
        if targetPropertyName is None:
            self.targetPropertyName = target.propertyName
        else:
            self.targetPropertyName = targetPropertyName
            
        try:
            self.targetReadOnly = target.isReadOnly()
        except AttributeError:
            self.targetReadOnly = False
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, self.targetPropertyName), 1, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        self.cChoice = wx.Choice(self, -1, size = [100,-1])
        self.cChoice.Bind(wx.EVT_CHOICE, self.onChange)
        hsizer.Add(self.cChoice, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        self.update()
        
        self.SetSizerAndFit(hsizer)
        
    def onChange(self, event=None):
        if not self.targetReadOnly:
            self.scope.frameWrangler.stop()
            self.target.setString(self.cChoice.GetStringSelection())
            self.scope.frameWrangler.start()
            self.parent.update()
        
    def update(self):
        choices = self.target.getAvailableValues()
        self.cChoice.SetItems(choices)
        self.cChoice.SetSelection(choices.index(self.target.getString()))
        
class BoolControl(wx.CheckBox):
    def __init__(self, parent, target):
        wx.CheckBox.__init__(self, parent, -1, target.propertyName)
        self.target = target
        self.scope = parent.scope
        self.parent = parent

        self.Bind(wx.EVT_CHECKBOX, self.onChange)        
        
        self.update()        
        
    def onChange(self, event=None):
        self.scope.frameWrangler.stop()
        self.target.setValue(self.GetValue())
        self.scope.frameWrangler.start()
        self.parent.update()
        
    def update(self):
        self.SetValue(self.target.getValue())
        
        

class ZylaControl(wx.Panel):
    def _init_ctrls(self):
        vsizer = wx.BoxSizer(wx.VERTICAL)

        for c in self.ctrls:
            vsizer.Add(c, 0, wx.EXPAND|wx.ALL, 2)
        
        self.SetSizerAndFit(vsizer)
        
    def __init__(self, parent, cam, scope):
        wx.Panel.__init__(self, parent)
        
        self.cam = cam
        self.scope = scope
        
        self.ctrls = [EnumControl(self, cam.SimplePreAmpGainControl, targetPropertyName='SimpleGainMode'),
                      EnumControl(self, cam.PixelReadoutRate),
                      BoolControl(self, cam.SpuriousNoiseFilter),
                      BoolControl(self, cam.StaticBlemishCorrection),
                      EnumControl(self, cam.CycleMode),
                      EnumControl(self, cam.TemperatureStatus),]

        if len(cam.TemperatureControl.getAvailableValues()) > 0: # need to check that Zyla has at least one mode
            self.ctrls.append(EnumControl(self, cam.TemperatureControl))

        self._init_ctrls()
        
    def update(self):
        for c in self.ctrls:
            c.update()

        
