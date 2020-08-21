# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:29:38 2020

@author: zacsimile
"""

import wx

# Built from PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel, but with a twist.


######## Base property controls
class PropertyControl:
    """
    Default for controlling camera properties. Restarts frameWrangler
    as settings are changed. Designed to be mixed with a wx element.

    Usage is

        MyControl(wx.SomeElement, PropertyControl):
            def Init(self, parent):
                # wx initialization
                # bind wx event to self.onChange, indicates when to update the camera property
                PropertyControl.Init(self, parent)

            def set_target_value(self):
                # Tell us how to set the value of self.target from the wx element

            def update(self):
                # Tell us how our wx.SomeElement should look, based on the value of self.target
    """
    def __init__(self, target):
        self.target = target  # String
    
    def Init(self, parent):
        self.parent=parent
        self.cam=self.parent.cam
        self.update()

    def onChange(self, event=None):
        self.parent.scope.frameWrangler.stop()
        self.set_target_value()
        self.parent.scope.frameWrangler.start()
        self.parent.update()

    def set_target_value(self):
        raise NotImplementedError('This function should be over-ridden in derived class')

    def update(self):
        raise NotImplementedError('This function should be over-ridden in derived class')

class BoolPropertyControl(wx.CheckBox, PropertyControl):
    """
    Property control for True/False values.
    """
    def Init(self, parent):
        wx.CheckBox.__init__(self, parent, -1, self.target)
        self.Bind(wx.EVT_CHECKBOX, self.onChange) 
        PropertyControl.Init(self, parent)

    def set_target_value(self):
        setattr(self.cam, self.target, self.GetValue())

    def update(self):
        self.SetValue(getattr(self.cam, self.target))


class EnumPropertyControl(wx.Panel, PropertyControl):
    """
    Property control for list of values.
    """
    def __init__(self, target, choices=None):
        PropertyControl.__init__(self, target)
        if type(choices) == str:
            # Assume camera property
            self._choices = getattr(self.cam, choices)
        elif type(choices) == list:
            self._choices = choices

    def Init(self, parent):
        wx.Panel.__init__(self, parent)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, self.target), 1, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        self.cChoice = wx.Choice(self, -1, size = [100,-1])
        self.cChoice.Bind(wx.EVT_CHOICE, self.onChange)
        hsizer.Add(self.cChoice, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.SetSizerAndFit(hsizer)
        PropertyControl.Init(self, parent)

    def set_target_value(self):
        setattr(self.cam, self.target, self.cChoice.GetSelection())

    def update(self):
        self.cChoice.SetItems(self._choices)
        self.cChoice.SetSelection(getattr(self.cam, self.target))

######## Property controls based on PYME.Acquire.Hardware.Camera class    
class ModePropertyControl(EnumPropertyControl):
    def __init__(self, target='_mode'):
        """
        List of available camera modes, grabbed from the Camera base class.
        """
        from PYME.Acquire.Hardware.Camera import Camera
        modes = [x for x in dir(Camera) if x.startswith('MODE_')]
        choices = sorted(modes, key=lambda x: getattr(Camera, x))
        EnumPropertyControl.__init__(self, target, choices)

    def set_target_value(self):
        self.cam.SetAcquisitionMode(self.cChoice.GetSelection())


######## Example replacement for legacy property control for ATBool
class ATBoolPropertyControl(BoolPropertyControl):
    """
    Property control for ATBools (see PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel).
    """
    def set_target_value(self):
        self.target.setValue(self.GetValue())

    def update(self):
        self.SetValue(self.target.getValue())

######## Camera control
class CameraControlFrame(wx.Panel):
    """
    Base class for camera controls. Usage is 


        ctrls = [BoolControl('SpuriousNoiseFilter'),
                etc.]
        MyCamControl = CamControl(parent, cam, scope, ctrls)

    """
    def _init_ctrls(self):
        vsizer = wx.BoxSizer(wx.VERTICAL)

        for c in self._ctrls:
            c.Init(self)
            vsizer.Add(c, 0, wx.EXPAND|wx.ALL, 2)
        
        self.SetSizerAndFit(vsizer)
        
    def __init__(self, parent, cam, scope, ctrls):
        wx.Panel.__init__(self, parent)
        
        self.cam = cam
        self.scope = scope

        self._ctrls = ctrls
        
        self._init_ctrls()
        
    def update(self):
        for c in self._ctrls:
            c.update()
