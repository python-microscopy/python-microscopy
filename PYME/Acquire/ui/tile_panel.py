import wx
from PYME.Acquire.Utils import tiler

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
    def __init__(self, parent, scope):
        wx.Panel.__init__(self, parent)
        
        self.scope=scope

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

    def OnQueue(self, event=None):
        import numpy as np
        from PYME.Acquire.ActionManager import UpdateState, SpoolSeries, FunctionAction
        from PYME.Acquire import protocol
        tile_protocols = [p for p in protocol.get_protocol_list() if 'tile' in p]

        dialog = wx.SingleChoiceDialog(self, '', 'Select Protocol', tile_protocols)

        ret = dialog.ShowModal()

        if ret != wx.ID_OK:
            dialog.Destroy()
            return
        
        protocol_name = dialog.GetStringSelection()
        dialog.Destroy()

        x_spacing = float(self.x_spacing_mm.GetValue()) * 1e3  # [mm -> um]
        y_spacing = float(self.y_spacing_mm.GetValue()) * 1e3  # [mm -> um]
        n_x = int(self.n_x.GetValue())
        n_y = int(self.n_y.GetValue())
        nice = int(self.nice.GetValue())

        curr_pos = self.scope.GetPos()

        # TODO - making this more flexible orientation wise, numbering for e.g.
        # 384wp, etc.. This puts H1 of a 96er at the min x, min y well.
        xind_names = np.array([chr(ord('@') + n) for n in range(1, n_x + 1)[::-1]])
        yind_names = np.arange(1, n_y + 1).astype(str)

        x_w = np.arange(0, n_x * x_spacing, x_spacing)
        y_w = np.arange(0, n_y * y_spacing, y_spacing)

        x_wells = []
        y_wells = np.repeat(y_w, n_x)
        names = []
        # zig-zag with turns along x
        for xi in range(n_y):
            if xi % 2:
                x_wells.extend(x_w[::-1])
                names.extend([xi_name + yind_names[xi] for xi_name in xind_names[::-1]])
            else:
                x_wells.extend(x_w)
                names.extend([xi_name + yind_names[xi] for xi_name in xind_names])
        x_wells = np.asarray(x_wells)

        # add the current scope position offset
        x_wells += curr_pos['x']
        y_wells += curr_pos['y']

        # queue them all
        actions = list()
        for x, y, filename in zip(x_wells, y_wells, names):
            state = UpdateState(state={'Positioning.x': x, 'Positioning.y': y})
            spool = SpoolSeries(protocol=protocol_name, stack=False, 
                                doPreflightCheck=False, fn=filename)
            actions.append(state.then(spool))
        
        actions.append(FunctionAction('turnAllLasersOff', {}))
        self.scope.actions.queue_actions(actions, nice)
        
