# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:55:50 2016

@author: david
"""
import wx
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ActionList(wx.ListCtrl):
    def __init__(self, parent, actionManager, pos=wx.DefaultPosition,
                 size=(480, 300), style=wx.LC_REPORT|wx.LC_VIRTUAL):
        wx.ListCtrl.__init__(self, parent, -1, pos, size, style)
        
        self.actionManager = actionManager
        self.actionManager.onQueueChange.connect(self.update)
        
        self.InsertColumn(0, "Priority")
        self.InsertColumn(1, "Action")
        #self.InsertColumn(2, "Args")
        self.InsertColumn(2, "Expiry")
        self.InsertColumn(3, 'Max Duration')
        
        self.SetColumnWidth(0, 50)
        self.SetColumnWidth(1, 600)
        #self.SetColumnWidth(2, 450)
        self.SetColumnWidth(2, 50)
        self.SetColumnWidth(3, 200)


    def OnGetItemText(self, item, col):
        vals = self._queueItems[item]
        
        val = vals[col]
        return repr(val)
        
    def update(self, **kwargs):
        self._queueItems = list(self.actionManager.actionQueue.queue)
        self._queueItems.sort(key=lambda a : a[0])
        self.SetItemCount(len(self._queueItems))
        self.Refresh()

ACTION_DEFAULTS = ['spoolController.StartSpooling',
                   'state.update',
                   ]

SORT_FUNCTIONS = {
    'None': lambda positions, scope_position: positions,
}

try:
    from PYME.Analysis.points.traveling_salesperson import sort as tspsort #avoid clobbering sort() builtin
    from PYME.Analysis.points.traveling_salesperson import queue_opt
    SORT_FUNCTIONS.update({'TSP': tspsort.tsp_sort, 'QTSP': queue_opt.TSPQueue})
except ImportError:
    pass

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
        self.tTimeout = wx.TextCtrl(self, -1, '1000000', size=(50, -1))
        hsizer.Add(self.tTimeout, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.Add(wx.StaticText(self, -1, 'Max Duration [s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.t_duration = wx.TextCtrl(self, -1, '%.1f' % np.finfo(float).max, size=(50, -1))
        hsizer.Add(self.t_duration, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
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

        hsizer.Add(wx.StaticText(self, -1, 'Sort Function:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        self.SortSelect = wx.ComboBox(self, -1, 'None', choices=list(SORT_FUNCTIONS.keys()), size=(150, -1))
        hsizer.Add(self.SortSelect, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)

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
        max_duration = float(self.t_duration.GetValue())
        self.actionManager.QueueAction(functionName, args, nice, timeout,
                                       max_duration)

    def OnAddMove(self, event):
        nice = float(self.tNice.GetValue())
        #functionName = self.tFunction.GetValue()
        #args = eval('dict(%s)' % self.tArgs.GetValue())

        functionName = 'state.update'
        args = {'state' : {'Positioning.x': self.scope.state['Positioning.x'], 
                           'Positioning.y': self.scope.state['Positioning.y'], 
                           'Positioning.z': self.scope.state['Positioning.z']}}

        timeout = float(self.tTimeout.GetValue())
        max_duration = float(self.t_duration.GetValue())
        self.actionManager.QueueAction(functionName, args, nice, timeout, 
                                       max_duration)
        

    def OnAddSequence(self, event):
        nice = float(self.tNice.GetValue())
        functionName = 'spoolController.StartSpooling'
        args = {'maxFrames' : int(self.tNumFrames.GetValue()), 'stack': bool(self.rbZStepped.GetValue())}
        timeout = float(self.tTimeout.GetValue())
        max_duration = float(self.t_duration.GetValue())
        self.actionManager.QueueAction(functionName, args, nice, timeout,
                                       max_duration)
        
    
    def _add_ROIs(self, rois):
        """
        Add ROI positioning and spooling actions to queue.

        Parameters
        ----------
        rois: list-like
            list of ROI (x, y) positions, or array of shape (n_roi, 2). Units in micrometers.
        
        Notes
        -----
        Currently ignores the `Max Duration` GUI control, ensuring the timeout
        is long enough for all queued ROI tasks, and with 10 s max duration on
        movements and ~2x acquisition time max durations for spooling series.
        """
        
        # coordinates are for the centre of ROI, and are referenced to the 0,0 pixel of the camera,
        # correct this for a custom ROI.
        roi_offset_x, roi_offset_y = self.scope.get_roi_offset()

        # subtract offset and reshape to N x 2 array
        positions = np.reshape(rois, (len(rois), 2)).astype(float) - np.array([roi_offset_x, roi_offset_y])[None, :]

        # apply sorting function
        scope_pos = self.scope.GetPos()
        positions = SORT_FUNCTIONS[self.SortSelect.GetValue()](positions, (scope_pos['x'], scope_pos['y']))

        # get queue parameters
        n_frames = int(self.tNumFrames.GetValue())
        nice = float(self.tNice.GetValue())
        time_est =  1.25 * n_frames / self.scope.cam.GetFPS()  # per series
        logger.debug('Expecting series to complete in %.1f s each' % time_est)
        # allow enough time for what we queue
        timeout = max(float(self.tTimeout.GetValue()), 
                      positions.shape[0] * time_est)
        for ri in range(positions.shape[0]):
            args = {'state': {'Positioning.x': positions[ri, 0], 'Positioning.y': positions[ri, 1]}}
            self.actionManager.QueueAction('state.update', args, nice, timeout, 10)
            args = {'maxFrames': n_frames, 'stack': bool(self.rbZStepped.GetValue())}
            self.actionManager.QueueAction('spoolController.StartSpooling', args, nice, timeout, 2 * time_est)
    
    def OnROIsFromFile(self, event):
        import wx
        from PYME.IO import tabular

        filename = wx.FileSelector("Load ROI Positions:", wildcard="*.hdf", flags=wx.FD_OPEN)
        if not filename == '':
            rois = tabular.HDFSource(filename, tablename='roi_locations')
            
            rois = [(x, y) for x, y in zip(rois['x_um'], rois['y_um'])]
            
            self._add_ROIs(rois)
    
    def OnROIsFromTileviewer(self, event):
        import requests
        resp = requests.get('http://localhost:8979/get_roi_locations')
        if resp.status_code != 200:
            raise requests.HTTPError('Could not get ROI locations')

        rois = np.array(resp.json())
        #print(rois.shape)
        self._add_ROIs(rois)
