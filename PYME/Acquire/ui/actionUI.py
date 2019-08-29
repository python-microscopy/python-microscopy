# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:55:50 2016

@author: david
"""
import wx
import numpy as np

class ActionList(wx.ListCtrl):
    def __init__(self, parent, actionManager, pos=wx.DefaultPosition,
                 size=(480, 300), style=wx.LC_REPORT|wx.LC_VIRTUAL):
        wx.ListCtrl.__init__(self, parent, -1, pos, size, style)
        
        self.actionManager = actionManager
        self.actionManager.onQueueChange.connect(self.update)
        
        self.InsertColumn(0, "Priority")
        self.InsertColumn(1, "ID")
        self.InsertColumn(2, "Function")
        self.InsertColumn(3, "Args")
        self.InsertColumn(4, "Expiry")
        
        self.SetColumnWidth(0, 50)
        self.SetColumnWidth(1, 15)
        self.SetColumnWidth(2, 150)
        self.SetColumnWidth(3, 450)
        self.SetColumnWidth(4, 50)


    def OnGetItemText(self, item, col):
        vals = self._queueItems[item]
        
        val = vals[col]
        return repr(val)
        
    def update(self, **kwargs):
        self._queueItems = list(self.actionManager.actionQueue.queue)
        self._queueItems.sort()
        self.SetItemCount(len(self._queueItems))
        self.Refresh()

ACTION_DEFAULTS = ['spoolController.StartSpooling',
                   'state.update',
                   ]

class ActionPanel(wx.Panel):
    def __init__(self, parent, actionManager, scope):
        wx.Panel.__init__(self, parent)
        self.actionManager = actionManager
        self.scope = scope

        vsizer = wx.BoxSizer(wx.VERTICAL)
        self.actionList = ActionList(self, self.actionManager)
        vsizer.Add(self.actionList, 1, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Nice:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tNice = wx.TextCtrl(self, -1, '10', size=(30, -1))
        hsizer.Add(self.tNice, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Function:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tFunction = wx.ComboBox(self, -1, '', choices=ACTION_DEFAULTS,size=(150, -1))
        hsizer.Add(self.tFunction, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Args:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tArgs = wx.TextCtrl(self, -1, '', size=(150, -1))
        hsizer.Add(self.tArgs, 1, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Timeout [s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tTimeout = wx.TextCtrl(self, -1, '1000000', size=(30, -1))
        hsizer.Add(self.tTimeout, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        self.bAdd = wx.Button(self, -1, 'Add', style=wx.BU_EXACTFIT)
        self.bAdd.Bind(wx.EVT_BUTTON, self.OnAddAction)
        hsizer.Add(self.bAdd, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        vsizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 4)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bMoveToHere = wx.Button(self, -1, 'Add move to current location')
        self.bMoveToHere.Bind(wx.EVT_BUTTON, self.OnAddMove)
        hsizer.Add(self.bMoveToHere, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.AddStretchSpacer()

        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.rbNoSteps = wx.RadioButton(self, -1, '2D', style=wx.RB_GROUP)
        hsizer.Add(self.rbNoSteps, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.rbZStepped = wx.RadioButton(self, -1, 'Z stepped')
        hsizer.Add(self.rbZStepped, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

        self.rbNoSteps.SetValue(True)
        
        hsizer.AddStretchSpacer()

        hsizer.Add(wx.StaticText(self, -1, 'Num frames: '), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.tNumFrames = wx.TextCtrl(self, -1, '10000', size=(50, -1))
        hsizer.Add(self.tNumFrames, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.AddStretchSpacer()

        self.bAddAquisition = wx.Button(self, -1, 'Add acquisition')
        self.bAddAquisition.Bind(wx.EVT_BUTTON, self.OnAddSequence)
        hsizer.Add(self.bAddAquisition, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bQueueROIsFromFile = wx.Button(self, -1, 'Queue ROIs from file')
        self.bQueueROIsFromFile.Bind(wx.EVT_BUTTON, self.OnROIsFromFile)
        hsizer.Add(self.bQueueROIsFromFile, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.bQueueROIsFromTileviewer = wx.Button(self, -1, 'Queue ROIs from Tile Viewer')
        self.bQueueROIsFromTileviewer.Bind(wx.EVT_BUTTON, self.OnROIsFromTileviewer)
        hsizer.Add(self.bQueueROIsFromTileviewer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)

        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        
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
    
    def OnAddAction(self, event):
        nice = float(self.tNice.GetValue())
        functionName = self.tFunction.GetValue()
        args = eval('dict(%s)' % self.tArgs.GetValue())
        timeout = float(self.tTimeout.GetValue())
        self.actionManager.QueueAction(functionName, args, nice, timeout)

    def OnAddMove(self, event):
        nice = float(self.tNice.GetValue())
        #functionName = self.tFunction.GetValue()
        #args = eval('dict(%s)' % self.tArgs.GetValue())

        functionName = 'state.update'
        args = {'state' : {'Positioning.x': self.scope.state['Positioning.x'], 
                           'Positioning.y': self.scope.state['Positioning.y'], 
                           'Positioning.z': self.scope.state['Positioning.z']}}

        timeout = float(self.tTimeout.GetValue())
        self.actionManager.QueueAction(functionName, args, nice, timeout)
        

    def OnAddSequence(self, event):
        nice = float(self.tNice.GetValue())
        functionName = 'spoolController.StartSpooling'
        args = {'maxFrames' : int(self.tNumFrames.GetValue()), 'stack': bool(self.rbZStepped.GetValue())}
        timeout = float(self.tTimeout.GetValue())
        self.actionManager.QueueAction(functionName, args, nice, timeout)
        
    
    def _add_ROIs(self, rois):
        """
        Add ROIs to queue, staggering their positioning and spooling actions

        Parameters
        ----------
        rois: list-like
            list of ROI (x, y) positions, or array of shape (n_roi, 2)

        Returns
        -------

        """
        priority_offset = 1.0 / (2 * len(rois))
        nice = float(self.tNice.GetValue())
        timeout = float(self.tTimeout.GetValue()) #CHECKME - default here might be too short
        
        for x, y in rois:
            args = {'state' : {'Positioning.x': float(x), 'Positioning.y': float(y)}}
            self.actionManager.QueueAction('state.update', args, nice, timeout)
            args = {'maxFrames': int(self.tNumFrames.GetValue()), 'stack': bool(self.rbZStepped.GetValue())}
            self.actionManager.QueueAction('spoolController.StartSpooling', args, nice + priority_offset, timeout)
            nice += 2 * priority_offset
    
    def OnROIsFromFile(self, event):
        import wx
        from PYME.IO import tabular

        filename = wx.FileSelector("Load ROI Positions:", wildcard="*.hdf", flags=wx.FD_OPEN)
        if not filename == '':
            rois = tabular.hdfSource(filename, tablename='roi_locations')
            
            rois = [(x, y) for x, y in zip(rois['x_um'], rois['y_um'])]
            
            self._add_ROIs(rois)
    
    def OnROIsFromTileviewer(self, event):
        import requests
        resp = requests.get('http://localhost:8979/get_roi_locations')
        if resp.status_code != 200:
            raise requests.HTTPError('Could not get ROI locations')

        rois = np.array(resp.json())
        print(rois.shape)
        self._add_ROIs(rois)
        


   