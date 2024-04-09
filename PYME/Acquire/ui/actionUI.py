# -*- coding: utf-8 -*-
"""
Created on Sat May 28 23:55:50 2016

@author: david
"""
import wx
import numpy as np
import logging
import time

from PYME.Acquire import actions
from PYME.ui import cascading_layout

logger = logging.getLogger(__name__)

class ActionList(wx.ListCtrl):
    def __init__(self, parent, actionManager, pos=wx.DefaultPosition,
                 size=(480, 300), style=wx.LC_REPORT|wx.LC_VIRTUAL):
        wx.ListCtrl.__init__(self, parent, -1, pos, size, style)
        
        self.actionManager = actionManager
        self.actionManager.onQueueChange.connect(self.update)
        
        self.InsertColumn(0, "When")
        self.InsertColumn(1, "Priority")
        self.InsertColumn(2, "Action")
        #self.InsertColumn(2, "Args")
        self.InsertColumn(3, "Expiry")
        self.InsertColumn(4, 'Max Duration')
        
        self.SetColumnWidth(0, 50)
        self.SetColumnWidth(1, 50)
        self.SetColumnWidth(2, 600)
        #self.SetColumnWidth(2, 450)
        self.SetColumnWidth(3, 50)
        self.SetColumnWidth(4, 200)


    def OnGetItemText(self, item, col):
        if item < len(self._queueItems):
            vals = self._queueItems[item]
            if (col==0): 
                return 'pending'
        else:
            item -= len(self._queueItems)
            when, vals = self._scheduledItems[item]
            if (col==0):
                return time.strftime('%H:%M:%S', time.localtime(when))
        
        val = vals[col-1]
        return repr(val)
        
    def update(self, **kwargs):
        self._queueItems = list(self.actionManager.actionQueue.queue)
        self._queueItems.sort(key=lambda a : a[0])
        self._scheduledItems = list(self.actionManager.scheduledQueue.queue)
        self._scheduledItems.sort(key=lambda a : a[0])
        self.SetItemCount(len(self._queueItems) + len(self._scheduledItems))
        self.Refresh()

ACTION_DEFAULTS = ['spoolController.start_spooling',
                   'state.update',
                   ]

SORT_FUNCTIONS = {
    'None': lambda positions, scope_position: positions,
}

ACTION_TYPES = ['MoveTo', 'UpdateState', 'CenterROIOn', 'SpoolSeries', 'FunctionAction']

try:
    from PYME.Analysis.points.traveling_salesperson import sort as tspsort #avoid clobbering sort() builtin
    from PYME.Analysis.points.traveling_salesperson import queue_opt
    SORT_FUNCTIONS.update({'TSP': tspsort.tsp_sort, 'QTSP': queue_opt.TSPQueue})
except ImportError:
    pass


class SingleActionPanel(wx.Panel, cascading_layout.CascadingLayoutMixin):
    description='An action that does something'
    supports_then =True

    def __init__(self, parent, actionManager, scope):
        wx.Panel.__init__(self, parent)
        self.actionManager = actionManager
        self.scope = scope

        self._pan_then = None
        
        sizer = wx.BoxSizer(wx.VERTICAL)

        self._init_controls(sizer)
        
        if self.supports_then:
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
            hsizer.Add(wx.StaticText(self, -1, 'Then:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
            self.cThen = wx.Choice(self, -1, choices=['None', ] + ACTION_TYPES)
            self.cThen.Bind(wx.EVT_CHOICE, self.OnThenChanged)
            hsizer.Add(self.cThen, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

            sizer.Add(hsizer, 0, wx.EXPAND, 0)

            self._then_sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self._then_sizer, 0, wx.EXPAND|wx.LEFT, 20)

        self.SetSizerAndFit(sizer)

    def _init_controls(self, sizer):
        pass

    def OnThenChanged(self, event):
        if self._pan_then is not None:
            self._pan_then.Destroy()
            self._then_sizer.Clear()
        
        then = self.cThen.GetStringSelection()
        if then != 'None':
            #print('Changing then to %s' % then)
            self._pan_then = globals()[then + 'Panel'](self, self.actionManager, self.scope)
            self._then_sizer.Add(self._pan_then, 0, wx.EXPAND, 0)
        else:
            self._pan_then = None

        #print('re-layouting')
        self.cascading_layout()
        

    def get_action(self):
        action = self._get_action()
        
        if self._pan_then:
            return action.then(self._pan_then.get_action())
        else:
            return action
        

    def _get_action(self):
        """Return an Action object that represents the current state of the panel
        
        This should be implemented in an action-specific subclass.
        """
        raise NotImplementedError('This should be implemented in a subclass')
                   
class MoveToPanel(SingleActionPanel):
    def _init_controls(self, sizer):
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'X:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tX = wx.TextCtrl(self, -1, '0', size=(50, -1))
        hsizer.Add(self.tX, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Y:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tY = wx.TextCtrl(self, -1, '0', size=(50, -1))
        hsizer.Add(self.tY, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.bSetCurrent = wx.Button(self, -1, 'Use current')
        self.bSetCurrent.Bind(wx.EVT_BUTTON, self.OnSetCurrent)
        hsizer.Add(self.bSetCurrent, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        sizer.Add(hsizer, 0, wx.EXPAND, 0)

    def OnSetCurrent(self, event):
        pos = self.scope.GetPos()
        self.tX.SetValue('%.2f' % pos['x'])
        self.tY.SetValue('%.2f' % pos['y'])

    def _get_action(self):
        x = float(self.tX.GetValue())
        y = float(self.tY.GetValue())
        return actions.MoveTo(x, y)

class UpdateStatePanel(SingleActionPanel):
    def _init_controls(self, sizer):
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'State:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tState = wx.TextCtrl(self, -1, '', size=(150, -1))
        hsizer.Add(self.tState, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        sizer.Add(hsizer, 0, wx.EXPAND, 0)

    def _get_action(self):
        # TODO - use a cleaner dictionary editor
        state = eval('dict(%s)' % self.tState.GetValue())
        return actions.UpdateState(state)

class CenterROIOnPanel(SingleActionPanel):
    def _init_controls(self, sizer):
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'X:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tX = wx.TextCtrl(self, -1, '0', size=(50, -1))
        hsizer.Add(self.tX, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Y:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tY = wx.TextCtrl(self, -1, '0', size=(50, -1))
        hsizer.Add(self.tY, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        sizer.Add(hsizer, 0, wx.EXPAND, 0)

    def _get_action(self):
        x = float(self.tX.GetValue())
        y = float(self.tY.GetValue())
        return actions.CentreROIOn(x, y)

class SpoolSeriesPanel(SingleActionPanel):
    supports_then = False

    def __init__(self, parent, actionManager, scope):
        super().__init__(parent, actionManager, scope)

        #scope.spoolController.onSettingsChange.connect(self._update)

    def _init_controls(self, sizer):
        self.stAqType = wx.StaticText(self, -1, 'Add an acquisition using the currently selected type and settings')
        sizer.Add(self.stAqType, 0, wx.ALL, 2)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stNumFramesLabel = wx.StaticText(self, -1, 'Max frames:')
        hsizer.Add(self.stNumFramesLabel, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tNumFrames = wx.TextCtrl(self, -1, '10000', size=(50, -1))
        hsizer.Add(self.tNumFrames, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        sizer.Add(hsizer, 0, wx.EXPAND, 0)

    def _get_action(self):
        settings = self.scope.spoolController.get_settings()
        settings['max_frames']  = int(self.tNumFrames.GetValue())
        return actions.SpoolSeries(settings=settings, preflight_mode='warn', )  

    def _update(self, **kwargs):
        aqType = self.scope.spoolController.acquisition_type
        if aqType == 'ProtocolAcquisition':
            self.stNumFramesLabel.Show()
            self.tNumFrames.Show()
        else:
            self.stNumFramesLabel.Hide()
            self.tNumFrames.Hide()

        self.stAqType.SetLabel(f'An {aqType} acquisition will be added with the following settings: {self.scope.spoolController.get_settings()}')

        self.cascading_layout()


class FunctionActionPanel(SingleActionPanel):
    supports_then = False

    def _init_controls(self, sizer):
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Function:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tFunction = wx.ComboBox(self, -1, '', choices=ACTION_DEFAULTS,size=(150, -1))
        hsizer.Add(self.tFunction, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.Add(wx.StaticText(self, -1, 'Args:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tArgs = wx.TextCtrl(self, -1, '', size=(150, -1))
        hsizer.Add(self.tArgs, 1, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

    def _get_action(self):
        function_name = self.tFunction.GetValue()
        args = eval('dict(%s)' % self.tArgs.GetValue())
        return actions.FunctionAction(function_name, args)


class ActionPanel(wx.Panel, cascading_layout.CascadingLayoutMixin):
    def __init__(self, parent, actionManager, scope):
        wx.Panel.__init__(self, parent)
        self.actionManager = actionManager
        self.scope = scope

        vsizer = wx.BoxSizer(wx.VERTICAL)
        self.actionList = ActionList(self, self.actionManager)
        vsizer.Add(self.actionList, 1, wx.EXPAND, 0)
        
        self.add_single_sizer = wx.StaticBoxSizer(wx.StaticBox(self, label='Add Single Action'), wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Action type:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.cActionType = wx.Choice(self, -1, choices=ACTION_TYPES)
        self.cActionType.Bind(wx.EVT_CHOICE, self.OnActionChanged)
        hsizer.Add(self.cActionType, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.add_single_sizer.Add(hsizer, 0, wx.EXPAND, 0)

        self._pan_action = MoveToPanel(self, self.actionManager, self.scope)
        self._pan_action_sizer = wx.BoxSizer(wx.VERTICAL)
        self._pan_action_sizer.Add(self._pan_action, 0, wx.EXPAND, 0)
        self.add_single_sizer.Add(self._pan_action_sizer, 0, wx.EXPAND|wx.LEFT, 20)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Delay[s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tDelay = wx.TextCtrl(self, -1, '0', size=(40, -1))
        hsizer.Add(self.tDelay, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Nice:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tNice = wx.TextCtrl(self, -1, '10', size=(30, -1))
        hsizer.Add(self.tNice, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        hsizer.Add(wx.StaticText(self, -1, 'Timeout[s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tTimeout = wx.TextCtrl(self, -1, '1000000', size=(50, -1))
        hsizer.Add(self.tTimeout, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.Add(wx.StaticText(self, -1, 'Max duration[s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.t_duration = wx.TextCtrl(self, -1, '%.1f' % np.finfo(float).max, size=(50, -1))
        hsizer.Add(self.t_duration, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.AddStretchSpacer()

        self.bAdd = wx.Button(self, -1, 'Add', style=wx.BU_EXACTFIT)
        self.bAdd.Bind(wx.EVT_BUTTON, self.OnAddAction)
        hsizer.Add(self.bAdd, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        self.add_single_sizer.Add(hsizer, 0, wx.EXPAND|wx.TOP, 10)
        vsizer.Add(self.add_single_sizer, 0, wx.EXPAND, 0)
       
        hsizer = wx.StaticBoxSizer(wx.StaticBox(self, label='Queue acquisitions for each ROI'), wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Sort Function:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        self.SortSelect = wx.ComboBox(self, -1, 'None', choices=list(SORT_FUNCTIONS.keys()), size=(150, -1))
        hsizer.Add(self.SortSelect, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)

        self.bQueueROIsFromFile = wx.Button(self, -1, 'from file')
        self.bQueueROIsFromFile.Bind(wx.EVT_BUTTON, self.OnROIsFromFile)
        hsizer.Add(self.bQueueROIsFromFile, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.bQueueROIsFromTileviewer = wx.Button(self, -1, 'from Tile Viewer')
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

    def OnActionChanged(self, event):
        if self._pan_action is not None:
            self._pan_action.Destroy()
            self._pan_action_sizer.Clear()
        
        action = self.cActionType.GetStringSelection()
        if action != 'None':
            #print('Changing action to %s' % action)
            self._pan_action = globals()[action + 'Panel'](self, self.actionManager, self.scope)
            self._pan_action_sizer.Add(self._pan_action, 0, wx.EXPAND, 0)
        else:
            self._pan_action = None
            logger.warning('No action selected (we shouldn\'t get here)')

        #print('re-layouting')
        self.cascading_layout()
        
    def OnPauseActions(self, event):
        if self.actionManager.paused:
            self.actionManager.paused = False
            self.bPause.SetLabel('Pause')
        else:
            self.actionManager.paused = True
            self.bPause.SetLabel('Resume')
    
    def OnAddAction(self, event):
        delay = float(self.tDelay.GetValue())
        nice = float(self.tNice.GetValue())
        #functionName = self.tFunction.GetValue()
        #args = eval('dict(%s)' % self.tArgs.GetValue())
        timeout = float(self.tTimeout.GetValue())
        max_duration = float(self.t_duration.GetValue())

        if delay > 0:
            execute_after = time.time() + delay
        else:
            execute_after = 0

        #self.actionManager.QueueAction(functionName, args, nice, timeout,
        #                               max_duration, execute_after=execute_after)
        self.actionManager.queue_actions([self._pan_action.get_action()], nice, timeout, max_duration, execute_after=execute_after)

    # def OnAddMove(self, event):
    #     nice = float(self.tNice.GetValue())
    #     timeout = float(self.tTimeout.GetValue())
    #     max_duration = float(self.t_duration.GetValue())

    #     state =  {'Positioning.x': self.scope.state['Positioning.x'],
    #               'Positioning.y': self.scope.state['Positioning.y'],
    #               'Positioning.z': self.scope.state['Positioning.z']}

    #     self.actionManager.queue_actions([actions.UpdateState(state),],
    #                                      nice, timeout, max_duration)
        

    # def OnAddSequence(self, event):
    #     nice = float(self.tNice.GetValue())
    #     timeout = float(self.tTimeout.GetValue())
    #     max_duration = float(self.t_duration.GetValue())

    #     settings = {'max_frames': int(self.tNumFrames.GetValue()), 'z_stepped': bool(self.rbZStepped.GetValue())}
        
    #     self.actionManager.queue_actions([actions.SpoolSeries(settings=settings, preflight_mode='warn'),],
    #                                      nice , timeout,max_duration)
        
    
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
        try:
            time_est =  1.25 * n_frames / self.scope.cam.GetFPS()  # per series
        except NotImplementedError:
            # specifically the simulated camera here, which has a non-predictable frame rate
            # use a conservative default of 10 s/frame (should not matter as simulation will generally not be doing 10s of thousands of series)
            time_est = 10*n_frames

        logger.debug('Expecting series to complete in %.1f s each' % time_est)
        # allow enough time for what we queue
        timeout = max(float(self.tTimeout.GetValue()), 
                      positions.shape[0] * time_est)
        
        acts = []
        for ri in range(positions.shape[0]):
            state = {'Positioning.x': positions[ri, 0], 'Positioning.y': positions[ri, 1]}
            settings = {'max_frames': n_frames, 'z_stepped': bool(self.rbZStepped.GetValue())}
            
            acts.append(actions.UpdateState(state).then(actions.SpoolSeries(settings=settings, preflight_mode='warn')))
            
        self.actionManager.queue_actions(acts, nice, timeout, 2 * time_est)
    
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
