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
from PYME.ui import progress

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
        
        self.SetColumnWidth(0, 60)
        self.SetColumnWidth(1, 50)
        self.SetColumnWidth(2, 600)
        #self.SetColumnWidth(2, 450)
        self.SetColumnWidth(3, 60)
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

        if col == 3: # expiry is a timestamp
            return time.strftime('%H:%M:%S', time.localtime(val))
        else:
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

ACTION_TYPES = ['MoveTo', 'UpdateState', 'CenterROIOn', 'SpoolSeries', 'RemoteSpoolSeries', 'SimultaneousSpoolSeries', 'FunctionAction']

try:
    from PYME.Analysis.points.traveling_salesperson import sort as tspsort #avoid clobbering sort() builtin
    from PYME.Analysis.points.traveling_salesperson import queue_opt
    SORT_FUNCTIONS.update({'TSP': tspsort.tsp_sort, 'QTSP': queue_opt.TSPQueue})
except ImportError:
    pass


class SingleActionPanel(wx.Panel, cascading_layout.CascadingLayoutMixin):
    description='An action that does something'
    supports_then =True
    num_actions = 1

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

            sizer.Add(hsizer, 0, wx.EXPAND|wx.TOP, 5)

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

    def _set_then(self, then):
        self.cThen.SetSelection(ACTION_TYPES.index(then)+1)
        self.OnThenChanged(None)
    
    def get_action(self, idx=0):
        action = self._get_action(idx)
        
        if self._pan_then:
            return action.then(self._pan_then.get_action(idx))
        else:
            return action
        
    def get_actions(self):
        return [self.get_action(i) for i in range(self.num_actions)]
        
    def _get_action(self, idx=0):
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

    def _get_action(self, idx=0):
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

    def _get_action(self, idx=0):
        # TODO - use a cleaner dictionary editor
        state = eval('dict(%s)' % self.tState.GetValue())
        return actions.UpdateState(state)

class CenterROIOnPanel(SingleActionPanel):
    def _init_controls(self, sizer):
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.rbModeSingle = wx.RadioButton(self, -1, 'Single ROI', style=wx.RB_GROUP)
        self.rbModeSingle.SetValue(True)
        self.rbModeSingle.Bind(wx.EVT_RADIOBUTTON, self.OnSetMode)
        hsizer.Add(self.rbModeSingle, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.rbModeROIList = wx.RadioButton(self, -1, 'ROI List')
        self.rbModeROIList.Bind(wx.EVT_RADIOBUTTON, self.OnSetMode)
        hsizer.Add(self.rbModeROIList, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        sizer.Add(hsizer, 0, wx.EXPAND, 0)
        
        # single mode
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stXLabel = wx.StaticText(self, -1, 'X:')
        hsizer.Add(self.stXLabel, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tX = wx.TextCtrl(self, -1, '0', size=(50, -1))
        hsizer.Add(self.tX, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        self.stYLabel = wx.StaticText(self, -1, 'Y:')
        hsizer.Add(self.stYLabel, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tY = wx.TextCtrl(self, -1, '0', size=(50, -1))
        hsizer.Add(self.tY, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.bSetCurrent = wx.Button(self, -1, 'Use current centre')
        self.bSetCurrent.Bind(wx.EVT_BUTTON, self.OnSetCurrent)
        hsizer.Add(self.bSetCurrent, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        sizer.Add(hsizer, 0, wx.EXPAND, 0)

        #ROI list mode
        self.stROIList = wx.StaticText(self, -1, 'No rois specified')
        sizer.Add(self.stROIList, 0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL) #wx.StaticBoxSizer(wx.StaticBox(self, label='Queue acquisitions for each ROI'), wx.HORIZONTAL)

        self.stSortFcnLabel = wx.StaticText(self, -1, 'Sort Function:')
        hsizer.Add(self.stSortFcnLabel, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        self.SortSelect = wx.ComboBox(self, -1, 'None', choices=list(SORT_FUNCTIONS.keys()), size=(150, -1))
        hsizer.Add(self.SortSelect, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)

        self.bQueueROIsFromFile = wx.Button(self, -1, 'from file')
        self.bQueueROIsFromFile.Bind(wx.EVT_BUTTON, self.OnROIsFromFile)
        hsizer.Add(self.bQueueROIsFromFile, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.bQueueROIsFromTileviewer = wx.Button(self, -1, 'from Tile Viewer')
        self.bQueueROIsFromTileviewer.Bind(wx.EVT_BUTTON, self.OnROIsFromTileviewer)
        hsizer.Add(self.bQueueROIsFromTileviewer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)

        sizer.Add(hsizer, 0, wx.EXPAND, 0)

        self.OnSetMode(None)

        wx.CallAfter(self._set_then, 'SpoolSeries')

    def OnSetCurrent(self, event):
        x, y = self.scope.get_roi_centre()
        self.tX.SetValue('%.2f' % x)
        self.tY.SetValue('%.2f' % y)

    def _get_action(self, idx=0):
        if self.rbModeSingle.GetValue():
            x = float(self.tX.GetValue())
            y = float(self.tY.GetValue())
            return actions.CentreROIOn(x, y)
        else:
            try:
                x, y = self._rois[idx, :]
            except AttributeError:
                raise ValueError('No ROIs have been loaded')
            return actions.CentreROIOn(x, y)
        
    @property
    def num_actions(self):
        if self.rbModeSingle.GetValue():
            return 1
        else:
            try:
                return len(self._rois)
            except AttributeError:
                raise ValueError('No ROIs have been loaded')
    
    def OnSetMode(self, event):
        if self.rbModeSingle.GetValue():
            self.stXLabel.Show()
            self.tX.Show()
            self.stYLabel.Show()
            self.tY.Show()
            self.bSetCurrent.Show()

            self.stROIList.Hide()
            self.stSortFcnLabel.Hide()
            self.SortSelect.Hide()
            self.bQueueROIsFromFile.Hide()
            self.bQueueROIsFromTileviewer.Hide()
        else:
            self.stXLabel.Hide()
            self.tX.Hide()
            self.stYLabel.Hide()
            self.tY.Hide()
            self.bSetCurrent.Hide()

            self.stROIList.Show()
            self.stSortFcnLabel.Show()
            self.SortSelect.Show()
            self.bQueueROIsFromFile.Show()
            self.bQueueROIsFromTileviewer.Show()

        wx.CallAfter(self.cascading_layout)

    def _add_ROIs(self, rois):
        positions = np.reshape(rois, (len(rois), 2)).astype(float)
        
        # apply sorting function
        scope_pos = self.scope.GetPos()
        positions = SORT_FUNCTIONS[self.SortSelect.GetValue()](positions, (scope_pos['x'], scope_pos['y']))

        self._rois = positions

        self.stROIList.SetLabel('Loaded %d ROIs' % len(self._rois))


    def OnROIsFromFile(self, event):
        # TODO - support .csv as well
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
        self._add_ROIs(rois)


        
        

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

    def _get_action(self, idx=0):
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

class RemoteSpoolSeriesPanel(SpoolSeriesPanel):
    supports_then = False

    def __init__(self, parent, actionManager, scope):
        super().__init__(parent, actionManager, scope)

        #scope.spoolController.onSettingsChange.connect(self._update)

    def _init_controls(self, sizer):
        self.stAqType = wx.StaticText(self, -1, 'Add a remote acquisition using the remotely selected type and settings')
        sizer.Add(self.stAqType, 0, wx.ALL, 2)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stRemoteInstanceName = wx.StaticText(self, -1, 'Remote instance name:')
        hsizer.Add(self.stRemoteInstanceName, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tRemoteInstanceName = wx.TextCtrl(self, -1, 'remote_acquire_instance', size=(150, -1))
        hsizer.Add(self.tRemoteInstanceName, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.stNumFramesLabel = wx.StaticText(self, -1, 'Max frames:')
        hsizer.Add(self.stNumFramesLabel, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tNumFrames = wx.TextCtrl(self, -1, '10000', size=(50, -1))
        hsizer.Add(self.tNumFrames, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        sizer.Add(hsizer, 0, wx.EXPAND, 0)

    def _get_action(self, idx=0):
        # use remote settings for acquisition type and type-specific settings
        remote_instance = self.tRemoteInstanceName.GetValue()
        settings = getattr(self.scope, remote_instance).spooling_info()['settings']

        # use local settings for spool method and compression
        settings.update(self.scope.spoolController.get_settings(method_only=True))
        settings['max_frames']  = int(self.tNumFrames.GetValue())

        return actions.RemoteSpoolSeries(remote_instance=remote_instance,settings=settings, preflight_mode='warn', ) 

class SimultaneousSpoolSeriesPanel(SpoolSeriesPanel):
    supports_then = False

    def __init__(self, parent, actionManager, scope):
        super().__init__(parent, actionManager, scope)

        #scope.spoolController.onSettingsChange.connect(self._update)

    def _init_controls(self, sizer):
        self.stAqType = wx.StaticText(self, -1, 'Add simultaneously executing remote and local acquisition')
        sizer.Add(self.stAqType, 0, wx.ALL, 2)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stRemoteInstanceName = wx.StaticText(self, -1, 'Remote instance name:')
        hsizer.Add(self.stRemoteInstanceName, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tRemoteInstanceName = wx.TextCtrl(self, -1, 'remote_acquire_instance', size=(150, -1))
        hsizer.Add(self.tRemoteInstanceName, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.stNumFramesLabel = wx.StaticText(self, -1, 'Max frames:')
        hsizer.Add(self.stNumFramesLabel, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tNumFrames = wx.TextCtrl(self, -1, '10000', size=(50, -1))
        hsizer.Add(self.tNumFrames, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        sizer.Add(hsizer, 0, wx.EXPAND, 0)

    def _get_action(self, idx=0):
        # use remote settings for acquisition type and type-specific settings
        remote_instance = self.tRemoteInstanceName.GetValue()
        remote_settings = getattr(self.scope, remote_instance).spooling_info()['settings']

        # use local settings for spool method and compression
        remote_settings.update(self.scope.spoolController.get_settings(method_only=True))
        remote_settings['max_frames']  = int(self.tNumFrames.GetValue())

        local_settings = self.scope.spoolController.get_settings()
        local_settings['max_frames']  = int(self.tNumFrames.GetValue())

        return actions.SimultaeneousSpoolSeries(remote_instance=remote_instance,local_settings=local_settings, remote_settings=remote_settings, preflight_mode='warn', )  
 

    # def _update(self, **kwargs):
    #     aqType = self.scope.spoolController.acquisition_type
    #     if aqType == 'ProtocolAcquisition':
    #         self.stNumFramesLabel.Show()
    #         self.tNumFrames.Show()
    #     else:
    #         self.stNumFramesLabel.Hide()
    #         self.tNumFrames.Hide()

    #     self.stAqType.SetLabel(f'An {aqType} acquisition will be added with the following settings: {self.scope.spoolController.get_settings()}')

    #     self.cascading_layout()


class FunctionActionPanel(SingleActionPanel):
    supports_then = False

    def _init_controls(self, sizer):
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Function:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tFunction = wx.ComboBox(self, -1, '', choices=ACTION_DEFAULTS,size=(150, -1))
        hsizer.Add(self.tFunction, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.Add(wx.StaticText(self, -1, 'Args:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tArgs = wx.TextCtrl(self, -1, '', size=(150, -1))
        hsizer.Add(self.tArgs, 1, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        sizer.Add(hsizer)

    def _get_action(self, idx=0):
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
        
        self.add_single_sizer = wx.StaticBoxSizer(wx.StaticBox(self, label='Add Action(s)'), wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Action type:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.cActionType = wx.Choice(self, -1, choices=ACTION_TYPES)
        self.cActionType.SetSelection(ACTION_TYPES.index('CenterROIOn'))
        self.cActionType.Bind(wx.EVT_CHOICE, self.OnActionChanged)
        hsizer.Add(self.cActionType, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.add_single_sizer.Add(hsizer, 0, wx.EXPAND, 0)

        self._pan_action = CenterROIOnPanel(self, self.actionManager, self.scope)
        self._pan_action_sizer = wx.BoxSizer(wx.VERTICAL)
        self._pan_action_sizer.Add(self._pan_action, 0, wx.EXPAND, 0)
        self.add_single_sizer.Add(self._pan_action_sizer, 0, wx.EXPAND|wx.LEFT, 20)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Delay[s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tDelay = wx.TextCtrl(self, -1, '0', size=(40, -1))
        hsizer.Add(self.tDelay, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        

        hsizer.Add(wx.StaticText(self, -1, 'Repetitions:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tRepeats = wx.TextCtrl(self, -1, '1', size=(30, -1))
        hsizer.Add(self.tRepeats, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.Add(wx.StaticText(self, -1, 'Period [s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tPeriod = wx.TextCtrl(self, -1, '0', size=(30, -1))
        hsizer.Add(self.tPeriod, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.Add(wx.StaticText(self, -1, 'Nice:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tNice = wx.TextCtrl(self, -1, '10', size=(30, -1))
        hsizer.Add(self.tNice, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        self.add_single_sizer.Add(hsizer, 0, wx.EXPAND|wx.TOP, 15)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Timeout [s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.tTimeout = wx.TextCtrl(self, -1, '1000000', size=(50, -1))
        hsizer.Add(self.tTimeout, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.Add(wx.StaticText(self, -1, 'Max duration [s]:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        self.t_duration = wx.TextCtrl(self, -1, '3600', size=(50, -1))
        hsizer.Add(self.t_duration, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)

        hsizer.AddStretchSpacer()

        self.bAdd = wx.Button(self, -1, 'Add', style=wx.BU_EXACTFIT)
        self.bAdd.Bind(wx.EVT_BUTTON, progress.managed(self.OnAddActions, self, 'Adding action'))
        hsizer.Add(self.bAdd, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        self.add_single_sizer.Add(hsizer, 0, wx.EXPAND|wx.TOP, 2)
        vsizer.Add(self.add_single_sizer, 0, wx.EXPAND, 0)

        
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
    

    def OnAddActions(self, event):
        delay = float(self.tDelay.GetValue())
        nice = float(self.tNice.GetValue())
        #functionName = self.tFunction.GetValue()
        #args = eval('dict(%s)' % self.tArgs.GetValue())
        timeout = float(self.tTimeout.GetValue())
        max_duration = float(self.t_duration.GetValue())

        repetitions = int(self.tRepeats.GetValue())
        period = float(self.tPeriod.GetValue())

        if delay > 0:
            execute_after = time.time() + delay
        else:
            execute_after = 0

        #self.actionManager.QueueAction(functionName, args, nice, timeout,
        #                               max_duration, execute_after=execute_after)'
        actions = self._pan_action.get_actions()

        # FIXME - this is a very empirical - maybe revisit
        t_est = actions[0].estimated_duration(self.scope)
        logger.debug('Expect actions to complete in %.1f s' % t_est)
        max_duration = max(2*t_est, max_duration)
        timeout = max(max_duration*len(actions), timeout)

        for i in range(repetitions):
            self.actionManager.queue_actions(actions, nice, timeout, max_duration, execute_after=execute_after)
            execute_after += period

    
    # def _add_ROIs(self, rois):
    #     """
    #     Add ROI positioning and spooling actions to queue.

    #     Parameters
    #     ----------
    #     rois: list-like
    #         list of ROI (x, y) positions, or array of shape (n_roi, 2). Units in micrometers.
        
    #     Notes
    #     -----
    #     Currently ignores the `Max Duration` GUI control, ensuring the timeout
    #     is long enough for all queued ROI tasks, and with 10 s max duration on
    #     movements and ~2x acquisition time max durations for spooling series.
    #     """
        
    #     # coordinates are for the centre of ROI, and are referenced to the 0,0 pixel of the camera,
    #     # correct this for a custom ROI.
    #     roi_offset_x, roi_offset_y = self.scope.get_roi_offset()

    #     # subtract offset and reshape to N x 2 array
    #     positions = np.reshape(rois, (len(rois), 2)).astype(float) - np.array([roi_offset_x, roi_offset_y])[None, :]

    #     # apply sorting function
    #     scope_pos = self.scope.GetPos()
    #     positions = SORT_FUNCTIONS[self.SortSelect.GetValue()](positions, (scope_pos['x'], scope_pos['y']))

    #     # get queue parameters
    #     n_frames = int(self.tNumFrames.GetValue())
    #     nice = float(self.tNice.GetValue())
    #     try:
    #         time_est =  1.25 * n_frames / self.scope.cam.GetFPS()  # per series
    #     except NotImplementedError:
    #         # specifically the simulated camera here, which has a non-predictable frame rate
    #         # use a conservative default of 10 s/frame (should not matter as simulation will generally not be doing 10s of thousands of series)
    #         time_est = 10*n_frames

    #     logger.debug('Expecting series to complete in %.1f s each' % time_est)
    #     # allow enough time for what we queue
    #     timeout = max(float(self.tTimeout.GetValue()), 
    #                   positions.shape[0] * time_est)
        
    #     acts = []
    #     for ri in range(positions.shape[0]):
    #         state = {'Positioning.x': positions[ri, 0], 'Positioning.y': positions[ri, 1]}
    #         settings = {'max_frames': n_frames, 'z_stepped': bool(self.rbZStepped.GetValue())}
            
    #         acts.append(actions.UpdateState(state).then(actions.SpoolSeries(settings=settings, preflight_mode='warn')))
            
    #     self.actionManager.queue_actions(acts, nice, timeout, 2 * time_est)
    
    
