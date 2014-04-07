#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

class FloatParam(object):
    def __init__(self, paramName, guiName, default=0, helpText=''):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default
        
    def createGUI(self, parent, mdh):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))
        self.tValue.SetValue('%3.2f' % mdh.getOrDefault(self.paramName, self.default))

        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = float(self.tValue.GetValue())
        

class IntParam(object):
    def __init__(self, paramName, guiName, default=0, helpText=''):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default
        
    def createGUI(self, parent, mdh):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))
        self.tValue.SetValue('%d' % mdh.getOrDefault(self.paramName, self.default))

        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = int(self.tValue.GetValue())
        
class StringParam(object):
    def __init__(self, paramName, guiName, default='', helpText=''):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default
        
    def createGUI(self, parent, mdh):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))
        self.tValue.SetValue(mdh.getOrDefault(self.paramName, self.default))

        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = self.tValue.GetValue()
        

class ChoiceParam(object):
    def __init__(self, paramName, guiName, default='', helpText=''):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default
        
    def createGUI(self, parent, mdh):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))
        self.tValue.SetValue(mdh.getOrDefault(self.paramName, self.default))

        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = int(self.tValue.GetValue())