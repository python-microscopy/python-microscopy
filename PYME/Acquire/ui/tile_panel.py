import wx
from PYME.Acquire.Utils import tiler
import logging


logger = logging.getLogger(__name__)


class TilePanel(wx.Panel):
    def __init__(self, parent, scope):
        wx.Panel.__init__(self, parent)
        
        self.scope=scope
        
        self._gui_proc = None
        
        vsizer=wx.BoxSizer(wx.VERTICAL)

        # hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer2.Add(wx.StaticText(pan, -1, 'Step Size x[mm]:'), 0, wx.ALL, 2)
        # self.tPixelSizeX = wx.TextCtrl(pan, -1, value='%3.4f' % ps.pixelsize[0])
        # hsizer2.Add(self.tPixelSizeX, 0, wx.ALL, 2)
        # vsizer.Add(hsizer2)
        #
        # hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer2.Add(wx.StaticText(pan, -1, 'Step Size y[mm]:'), 0, wx.ALL, 2)
        # self.tPixelSizeY = wx.TextCtrl(pan, -1, value='%3.4f' % ps.pixelsize[1])
        # hsizer2.Add(self.tPixelSizeY, 0, wx.ALL, 2)
        # vsizer.Add(hsizer2)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(self, -1, '# x steps:'), 0, wx.ALL, 2)
        self.tXTiles = wx.TextCtrl(self, -1, value='%d' % 10)
        hsizer2.Add(self.tXTiles, 0, wx.ALL, 2)
        vsizer.Add(hsizer2)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(self, -1, '# y steps:'), 0, wx.ALL, 2)
        self.tYTiles = wx.TextCtrl(self, -1, value='%d' % 10)
        hsizer2.Add(self.tYTiles, 0, wx.ALL, 2)
        vsizer.Add(hsizer2)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(self, -1, 'Save to:'), 0, wx.ALL, 2)
        self.tDestination = wx.TextCtrl(self, -1, value='')
        hsizer2.Add(self.tDestination, 1, wx.ALL|wx.EXPAND, 2)
        vsizer.Add(hsizer2, 0, wx.EXPAND, 0)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.pProgress = wx.Gauge(self, -1, range=100)
        hsizer2.Add(self.pProgress, 1, wx.ALL|wx.EXPAND, 2)
        vsizer.Add(hsizer2, 0, wx.EXPAND, 0)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        #self.bTest = wx.Button(self, -1, 'Test')
        #self.bTest.Bind(wx.EVT_BUTTON, self.OnTest)
        #self.bTest.Disable()
        #hsizer2.Add(self.bTest, 0, wx.ALL, 2)
        self.bGo = wx.Button(self, -1, 'Go')
        self.bGo.Bind(wx.EVT_BUTTON, self.OnGo)
        hsizer2.Add(self.bGo, 0, wx.ALL, 2)
        self.bStop = wx.Button(self, -1, 'Stop')
        self.bStop.Disable()
        self.bStop.Bind(wx.EVT_BUTTON, self.OnStop)
        hsizer2.Add(self.bStop, 0, wx.ALL, 2)
        vsizer.Add(hsizer2)
        
        self.SetSizerAndFit(vsizer)
        
    def OnGo(self, event=None):
        # run a triggered tile acquisition if the camera is capable
        # FIXME - the hasattr test becomes problematic once we add FireSoftwareTrigger to our base camera class (to
        # document API)
        trigger = hasattr(self.scope.cam, 'FireSoftwareTrigger')
        
        self.scope.tiler = tiler.Tiler(self.scope, tile_dir = self.tDestination.GetValue(),
                                       n_tiles=(int(self.tXTiles.GetValue()), int(self.tYTiles.GetValue())),
                                       trigger=trigger)
        
        self.bStop.Enable()
        self.bGo.Disable()
        
        self.scope.tiler.on_stop.connect(self._on_stop)
        self.scope.tiler.progress.connect(self._update)
        self.scope.tiler.start()
        
        
    def OnStop(self, event=None):
        self.scope.tiler.stop()
        
    def _update(self, *args, **kwargs):
        self.pProgress.SetValue(100*float(self.scope.tiler.callNum)/self.scope.tiler.imsize)
        
    def _on_stop(self, *args, **kwargs):
        self.bStop.Disable()
        self.bGo.Enable()
        
        self.scope.tiler.on_stop.disconnect(self._on_stop)
        self.scope.tiler.progress.disconnect(self._update)
        
        # FIXME - previous delay was 1e3, which seems more reasonable. Do we need a config option (or heuristic) here ?
        # assume this change was due to the time it takes to build a pyramid after tiling ends. Might ultimately be fixed when we revisit live tiling. 
        wx.CallAfter(wx.CallLater,1e4, self._launch_viewer)
        
        
    def _launch_viewer(self):
        import subprocess
        import sys
        import webbrowser
        import time
        import requests
        import os

        self.scope.tiler.P.update_pyramid()
        
        #if not self._gui_proc is None:
        #    self._gui_proc.kill()

        # abs path the tile dir
        tiledir = self.scope.tiler._tiledir
        if not os.path.isabs(tiledir):
            # TODO - should we be doing the `.isabs()` check on the parent directory instead?
            from PYME.IO.FileUtils import nameUtils
            tiledir = nameUtils.getFullFilename(tiledir)
        
        try:  # if we already have a tileviewer serving, change the directory
            requests.get('http://127.0.0.1:8979/set_tile_source?tile_dir=%s' % tiledir)
        except requests.ConnectionError:  # start a new process
            try:
                pargs = {'creationflags': subprocess.CREATE_NEW_CONSOLE}
            except AttributeError:  # not on windows
                pargs = {'shell': True}
            
            self._gui_proc = subprocess.Popen('%s -m PYME.tileviewer.tileviewer %s' % (sys.executable, tiledir), **pargs)
            time.sleep(3)
            
        webbrowser.open('http://127.0.0.1:8979/')

class CircularTilePanel(TilePanel):
    def __init__(self, parent, scope):
        wx.Panel.__init__(self, parent)

        self.scope = scope

        self._gui_proc = None

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(self, -1, u'Scan radius [\u03BCm]:'), 0, wx.ALL, 2)
        self.radius_um = wx.TextCtrl(self, -1, value='%.1f' % 250)
        hsizer2.Add(self.radius_um, 0, wx.ALL, 2)
        vsizer.Add(hsizer2)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.return_home_checkbox = wx.CheckBox(self, -1, 'Return to start on completion')
        hsizer2.Add(self.return_home_checkbox, 0, wx.ALL, 2)
        vsizer.Add(hsizer2)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        hsizer2.Add(wx.StaticText(self, -1, 'Save to:'), 0, wx.ALL, 2)
        self.tDestination = wx.TextCtrl(self, -1, value='')
        hsizer2.Add(self.tDestination, 1, wx.ALL | wx.EXPAND, 2)
        vsizer.Add(hsizer2, 0, wx.EXPAND, 0)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.pProgress = wx.Gauge(self, -1, range=100)
        hsizer2.Add(self.pProgress, 1, wx.ALL | wx.EXPAND, 2)
        vsizer.Add(hsizer2, 0, wx.EXPAND, 0)

        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)
        # self.bTest = wx.Button(self, -1, 'Test')
        # self.bTest.Bind(wx.EVT_BUTTON, self.OnTest)
        # self.bTest.Disable()
        # hsizer2.Add(self.bTest, 0, wx.ALL, 2)
        self.bGo = wx.Button(self, -1, 'Go')
        self.bGo.Bind(wx.EVT_BUTTON, self.OnGo)
        hsizer2.Add(self.bGo, 0, wx.ALL, 2)
        self.bStop = wx.Button(self, -1, 'Stop')
        self.bStop.Disable()
        self.bStop.Bind(wx.EVT_BUTTON, self.OnStop)
        hsizer2.Add(self.bStop, 0, wx.ALL, 2)
        vsizer.Add(hsizer2)

        self.SetSizerAndFit(vsizer)

    def OnGo(self, event=None):
        trigger = hasattr(self.scope.cam, 'FireSoftwareTrigger')

        self.scope.tiler = tiler.CircularTiler(self.scope, tile_dir=self.tDestination.GetValue(),
                                               max_radius_um=float(self.radius_um.GetValue()), trigger=trigger,
                                               return_to_start=self.return_home_checkbox.GetValue())

        self.bStop.Enable()
        self.bGo.Disable()

        self.scope.tiler.on_stop.connect(self._on_stop)
        self.scope.tiler.progress.connect(self._update)
        self.scope.tiler.start()


class MultiwellTilePanel(TilePanel):
    def __init__(self, parent, scope):
        wx.Panel.__init__(self, parent)

        self.scope = scope

        self._gui_proc = None

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Save to:'), 0, wx.ALL, 2)
        self.tDestination = wx.TextCtrl(self, -1, value='')
        hsizer.Add(self.tDestination, 1, wx.ALL | wx.EXPAND, 2)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, u'Well Scan radius [\u03BCm]:'), 0, wx.ALL, 2)
        self.radius_um = wx.TextCtrl(self, -1, value='%.1f' % 250)
        hsizer.Add(self.radius_um, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, '# wells x:'), 0, wx.ALL, 2)
        self.n_x = wx.TextCtrl(self, -1, value='%d' % 3)
        hsizer.Add(self.n_x, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'x cent. dist [mm]:'), 0, wx.ALL, 2)
        self.x_spacing_mm = wx.TextCtrl(self, -1, value='%.1f' % 9)
        hsizer.Add(self.x_spacing_mm, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, '# wells y:'), 0, wx.ALL, 2)
        self.n_y = wx.TextCtrl(self, -1, value='%d' % 3)
        hsizer.Add(self.n_y, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'y cent. dist [mm]:'), 0, wx.ALL, 2)
        self.y_spacing_mm = wx.TextCtrl(self, -1, value='%.1f' % 9)
        hsizer.Add(self.y_spacing_mm, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bGo = wx.Button(self, -1, 'Go')
        # self.bGo.Disable()
        self.bGo.Bind(wx.EVT_BUTTON, self.OnGo)
        hsizer.Add(self.bGo, 0, wx.ALL, 2)
        self.bStop = wx.Button(self, -1, 'Stop')
        self.bStop.Disable()
        self.bStop.Bind(wx.EVT_BUTTON, self.OnStop)
        hsizer.Add(self.bStop, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        self.SetSizerAndFit(vsizer)


    def OnGo(self, event=None):
        trigger = hasattr(self.scope.cam, 'FireSoftwareTrigger')

        # get laser states
        laser_state = {k:v for k, v in dict(self.scope.state).items() if k.startswith('Laser')}

        self.scope.tiler = tiler.MultiwellCircularTiler(float(self.radius_um.GetValue()),
            float(self.x_spacing_mm.GetValue()) * 1e3, float(self.y_spacing_mm.GetValue()) * 1e3,
            int(self.n_x.GetValue()), int(self.n_y.GetValue()), self.scope, self.tDestination.GetValue(),
            trigger=trigger, laser_state=laser_state)

        self.bStop.Enable()
        self.bGo.Disable()

        # self.scope.tiler.on_stop.connect(self._on_stop)
        # self.scope.tiler.progress.connect(self._update)
        self.scope.tiler.start()

    def OnStop(self, event=None):
        self.scope.tiler.stop()


class MultiwellProtocolQueuePanel(wx.Panel):
    # TODO - refactor into Acquire.htsms - this is aquiring a bit of cruft that we probably don't want to support in the long term
    def __init__(self, parent, scope):
        wx.Panel.__init__(self, parent)
        
        self.scope=scope
        self._shame_index = 0
        self.scope.multiwellpanel = self
        self._drop_wells = []
        self.fast_axis = 'y'

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, '# wells x:'), 0, wx.ALL, 2)
        self.n_x = wx.TextCtrl(self, -1, value='%d' % 8)
        hsizer.Add(self.n_x, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'x cent. dist [mm]:'), 0, wx.ALL, 2)
        self.x_spacing_mm = wx.TextCtrl(self, -1, value='%.1f' % 9)
        hsizer.Add(self.x_spacing_mm, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, '# wells y:'), 0, wx.ALL, 2)
        self.n_y = wx.TextCtrl(self, -1, value='%d' % 12)
        hsizer.Add(self.n_y, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'y cent. dist [mm]:'), 0, wx.ALL, 2)
        self.y_spacing_mm = wx.TextCtrl(self, -1, value='%.1f' % 9)
        hsizer.Add(self.y_spacing_mm, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Nice:'), 0, wx.ALL, 2)
        self.nice = wx.TextCtrl(self, -1, value='%d' % 11)
        hsizer.Add(self.nice, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.axis_select_box = wx.ComboBox(self, -1, choices=['x', 'y'],
                                           value='y', size=(65, -1),
                                           style=wx.CB_DROPDOWN | wx.TE_PROCESS_ENTER)

        hsizer.Add(self.axis_select_box, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.axis_select_box.Bind(wx.EVT_COMBOBOX, self._on_combo_box)
        
        self.cb_get_it_done = wx.CheckBox(self, -1, 'Requeue missed')
        hsizer.Add(self.cb_get_it_done, 0, wx.ALL, 2)
        hsizer.Fit(self)
        vsizer.Add(hsizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.queue_button = wx.Button(self, -1, 'Queue')
        # self.bGo.Disable()
        self.queue_button.Bind(wx.EVT_BUTTON, self.OnQueue)
        hsizer.Add(self.queue_button, 0, wx.ALL, 2)
        # self.bStop = wx.Button(self, -1, 'Stop')
        # self.bStop.Disable()
        # self.bStop.Bind(wx.EVT_BUTTON, self.OnStop)
        # hsizer.Add(self.bStop, 0, wx.ALL, 2)
        vsizer.Add(hsizer)

        self.SetSizerAndFit(vsizer)
    
    def _on_combo_box(self, event):
        cb = event.GetEventObject()
        self.fast_axis = cb.GetValue()
    
    def requeue_missed(self, n_x, n_y, x_spacing, y_spacing, start_pos, protocol_name, nice=20, sleep=5):
        from PYME.Acquire.actions import FunctionAction
        from PYME.IO import clusterIO
        import posixpath
        import time

        logger.debug('requeuing missed wells')
        time.sleep(sleep)

        spooldir = self.scope.spoolController.dirname
        detections_pattern = posixpath.join(spooldir, '[A-Z][0-9]*_detections.h5')
        imaged = clusterIO.cglob(detections_pattern)
        imaged_wells = [im.split('/')[-1].split('_detections.h5')[0] for im in imaged]
        logger.debug('imaged %d wells' % len(imaged_wells))

        x_wells, y_wells, names = self._get_positions(n_x, n_y, x_spacing, y_spacing, start_pos)
        x_wells, y_wells, names = self._pop_wells(x_wells, y_wells, names, self._drop_wells)
        to_pop = [fn.split('_')[0] for fn in imaged_wells]
        x_wells, y_wells, names = self._pop_wells(x_wells, y_wells, names, to_pop)

        if len(names) < 1:
            return
        
        self._shame_index += 1
        shame_suffix = '_%d' % self._shame_index
        names = [name + shame_suffix for name in names]
        actions = self._get_action_list(x_wells, y_wells, names, protocol_name)
        
        actions.append(FunctionAction('turnAllLasersOff', {}))

        # lets just make it recursive for fun
        if self.cb_get_it_done.GetValue():
            actions.append(FunctionAction('multiwellpanel.requeue_missed', 
                                          {'n_x': n_x, 'n_y': n_y, 
                                          'x_spacing': x_spacing, 'y_spacing': y_spacing, 
                                          'start_pos': start_pos, 
                                          'protocol_name': protocol_name,
                                          'nice': nice}))
        
        logger.debug('requeuing %d wells' % len(names))
        self.scope.actions.queue_actions(actions, nice)

    def OnQueue(self, event=None):
        from PYME.Acquire.actions import FunctionAction
        from PYME.Acquire import protocol
        tile_protocols = [p for p in protocol.get_protocol_list() if 'tile' in p]

        dialog = wx.SingleChoiceDialog(self, '', 'Select Protocol', tile_protocols)

        ret = dialog.ShowModal()

        if ret != wx.ID_OK:
            dialog.Destroy()
            return
        
        protocol_name = dialog.GetStringSelection()
        dialog.Destroy()

        self._shame_index = 0

        x_spacing = float(self.x_spacing_mm.GetValue()) * 1e3  # [mm -> um]
        y_spacing = float(self.y_spacing_mm.GetValue()) * 1e3  # [mm -> um]
        n_x = int(self.n_x.GetValue())
        n_y = int(self.n_y.GetValue())
        nice = int(self.nice.GetValue())

        curr_pos = self.scope.GetPos()

        x_wells, y_wells, names = self._get_positions(n_x, n_y, x_spacing, y_spacing, curr_pos)
        x_wells, y_wells, names = self._pop_wells(x_wells, y_wells, names, self._drop_wells)
        actions = self._get_action_list(x_wells, y_wells, names, protocol_name)
        
        actions.append(FunctionAction('turnAllLasersOff', {}))

        if self.cb_get_it_done.GetValue():
            actions.append(FunctionAction('multiwellpanel.requeue_missed', 
                                          {'n_x': n_x, 'n_y': n_y,
                                           'x_spacing': x_spacing, 'y_spacing': y_spacing,
                                           'start_pos': curr_pos, 
                                           'protocol_name': protocol_name,
                                           'nice': nice}))

        self.scope.actions.queue_actions(actions, nice)
    
    def _get_positions(self, n_x, n_y, x_spacing, y_spacing, start_pos):
        import numpy as np
        # TODO - making this more flexible orientation wise, numbering for e.g.
        # 384wp, etc.. This puts H1 of a 96er at the min x, min y well.
        xind_names = np.array([chr(ord('@') + n) for n in range(1, n_x + 1)[::-1]])
        yind_names = np.arange(1, n_y + 1).astype(str)

        x_w = np.arange(0, n_x * x_spacing, x_spacing)
        y_w = np.arange(0, n_y * y_spacing, y_spacing)

        
        if self.fast_axis == 'x':
            # zig-zag with turns along x
            x_wells = []
            y_wells = np.repeat(y_w, n_x)
            names = []
            for xi in range(n_y):
                if xi % 2:
                    x_wells.extend(x_w[::-1])
                    names.extend([xi_name + yind_names[xi] for xi_name in xind_names[::-1]])
                else:
                    x_wells.extend(x_w)
                    names.extend([xi_name + yind_names[xi] for xi_name in xind_names])
            x_wells = np.asarray(x_wells)
        else:
            # zig-zag with turns along y
            y_wells = []
            x_wells = np.repeat(x_w, n_y)
            names = []
            for yi in range(n_x):
                if yi % 2:
                    y_wells.extend(y_w[::-1])
                    names.extend([xind_names[yi] + yi_name for yi_name in yind_names[::-1]])
                else:
                    y_wells.extend(y_w)
                    names.extend([xind_names[yi] + yi_name for yi_name in yind_names])
            y_wells = np.asarray(y_wells)

        # add the current scope position offset
        x_wells += start_pos['x']
        y_wells += start_pos['y']

        return x_wells, y_wells, names
        
    
    def _get_action_list(self, x_wells, y_wells, names, protocol_name):
        from PYME.Acquire.actions import UpdateState, SpoolSeries
        actions = list()
        for x, y, filename in zip(x_wells, y_wells, names):
            state = UpdateState(state={'Positioning.x': x, 'Positioning.y': y})
            spool = SpoolSeries(protocol=protocol_name, stack=False, 
                                doPreflightCheck=False, fn=filename)
            actions.append(state.then(spool))
        return actions
    
    def _pop_wells(self, x_wells, y_wells, names, to_pop):
        import numpy as np

        pop_inds = []
        for well in to_pop:
            try:
                pop_inds.append(names.index(well))
            except ValueError:
                pass
        
        if len(pop_inds) < 1:
            return x_wells, y_wells, names
        x_wells = x_wells.tolist()
        y_wells = y_wells.tolist()

        for pfn in sorted(pop_inds)[::-1]:
            x_wells.pop(pfn)
            y_wells.pop(pfn)
            names.pop(pfn)
        
        return np.asarray(x_wells), np.asarray(y_wells), names
