# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:55:50 2016

@author: david
"""
import wx

class ActionList(wx.ListCtrl):
    def __init__(self, parent, actionManager, pos=wx.DefaultPosition,
                 size=(480, 300), style=wx.LC_REPORT|wx.LC_VIRTUAL):
        wx.ListCtrl.__init__(self, parent, -1, pos, size, style)
        
        self.actionManager = actionManager
        self.actionManager.onQueueChange.connect(self.update)
        
        self.InsertColumn(0, "Priority")
        self.InsertColumn(1, "Function")
        self.InsertColumn(2, "Args")
        self.InsertColumn(3, "Expiry")
        
        self.SetColumnWidth(0, 50)
        self.SetColumnWidth(1, 250)
        self.SetColumnWidth(2, 500)
        self.SetColumnWidth(3, 150)


    def OnGetItemText(self, item, col):
        vals = self.actionManager.actionQueue.queue[item]
        
        val = vals[col]
        return repr(val)
        
    def update(self, **kwargs):
        self.SetItemCount(self.actionManager.actionQueue.qsize())
        self.Refresh()


class ActionPanel(wx.Panel):
    def __init__(self, parent, actionManager):#, scope):
        wx.Panel.__init__(self, parent)
        self.actionManager = actionManager

        vsizer = wx.BoxSizer(wx.VERTICAL)
        self.actionList = ActionList(self, self.actionManager)
        vsizer.Add(self.actionList, 1, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bPause = wx.Button(self, -1, 'Pause')
        self.bPause.Bind(wx.EVT_BUTTON, self.OnPauseActions)
        
        hsizer.AddStretchSpacer()
        hsizer.Add(self.bPause, 0, wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        
        #hsizer.Add(vsizer, 0, 0, 0)

        self.SetSizerAndFit(vsizer)
        
    def OnPauseActions(self, event):
        if self.actionManager.paused:
            self.actionManager.paused = False
            self.bPause.SetLabel('Pause')
        else:
            self.actionManager.paused = True
            self.bPause.SetLabel('Resume')

   