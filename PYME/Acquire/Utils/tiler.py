from . import pointScanner
from PYME.Analysis import tile_pyramid
import numpy as np
import time
from PYME.IO import MetaDataHandler
import os
import uuid
import datetime
from PYME.contrib import dispatch
import logging
logger = logging.getLogger(__name__)

from PYME.Acquire.acquisition_base import AcquisitionBase
from PYME.IO import acquisition_backends
from PYME.Acquire import eventLog, microscope

class TileAcquisition(AcquisitionBase):
    FILE_EXTENSION = '.tiles' # TODO - make .zarr compatable and use .zarr instead
    
    def __init__(self, scope : microscope.Microscope, tile_dir : str, n_tiles = 10, tile_spacing=None, dwelltime = 1, background=0, evtLog=None,
                 trigger='auto', base_tile_size=256, return_to_start=True, save_raw=False, backend='file', backend_kwargs={}):
        """
        :param return_to_start: bool
            Flag to toggle returning home at the end of the scan. False leaves scope position as-is on scan completion.
        """
        
        if trigger == 'auto':
            trigger = scope.cam.supports_software_trigger
        self._trigger = trigger

        self._return_to_start = return_to_start

        if evtLog is None:
            if save_raw:
                evtLog = True
            else:
                evtLog = False
        
        fs = np.array(scope.frameWrangler.currentFrame.shape[:2])
        
        if tile_spacing is None:
            #calculate tile spacing such that there is 20% overlap.
            tile_spacing = 0.8
        
        tile_spacing_um = tile_spacing*fs*np.array(scope.GetPixelSize())
            
        # pointScanner.PointScanner.__init__(self, scope=scope, pixels=n_tiles, pixelsize=tile_spacing_um,
        #                                    dwelltime=dwelltime, background=background, avg=False, evtLog=evtLog,
        #                                    trigger=trigger, stop_on_complete=True, return_to_start=return_to_start)
        
        self._scanner = pointScanner.Scanner(scope, pixels=n_tiles, pixelsize=tile_spacing_um, evtLog=evtLog, stop_on_complete=True)
        self.scope = scope
        
        self._tiledir = tile_dir
        self._base_tile_size = base_tile_size
        
        self._background = background
        self._flat = None #currently not used

        self._uuid = uuid.uuid4() # for dispatch
        
        self._last_update_time = 0

        if backend is acquisition_backends.ClusterBackend:
            self._backend_type='cluster'
        elif backend is acquisition_backends.HDFBackend:
            self._backend_type='file'
        else:
            raise ValueError('Unknown backend')

        # save the raw frames as well as the constructed pyramid - useful if we want to do something
        # fancy like cross-correlation based alignment of raw frames before stitching
        self._save_raw = save_raw
        if save_raw:
            if (self._backend_type == 'cluster') and not backend_kwargs.get('cluster_h5', False):
                fn = 'raw_frames.pcs'
            else:
                fn = 'raw_frames.h5'

            
            os.makedirs(tile_dir, exist_ok=True)
            self.storage = backend(os.path.join(tile_dir, fn), **backend_kwargs)
        
        
        #self.on_stop = dispatch.Signal()
        #self.on_progress = dispatch.Signal()
        AcquisitionBase.__init__(self)
        
    @classmethod
    def from_spool_settings(cls, scope, settings, backend, backend_kwargs={}, series_name=None, spool_controller=None):
        '''Create an Acquisition object from settings and a backend.'''

        tiling_settings = {
            'tile_dir': series_name,
        }

        tiling_settings.update(settings.get('tiling_settings', scope.tile_settings))
        
        #fix timing when using fake camera
        #TODO - move logic into backend?
        if scope.cam.__class__.__name__ == 'FakeCamera':
            backend_kwargs['spoof_timestamps'] = True
            backend_kwargs['cycle_time'] = scope.cam.GetIntegTime()
        
        return cls(scope=scope, backend=backend, backend_kwargs=backend_kwargs, **tiling_settings)
    
    @classmethod
    def get_frozen_settings(cls, scope, spool_controller=None):
        return {'tiling_settings': getattr(scope, 'tile_settings', {})}
    
    @classmethod
    def get_tiled_area(cls, scope, settings):
        fs = np.array(scope.frameWrangler.currentFrame.shape[:2])
        #calculate tile spacing such that there is 20% overlap.
        tile_spacing = settings.get(0.8*fs*np.array(scope.GetPixelSize()))

        nx, ny = settings.get('n_tiles', (10, 10))
        return nx * fs[0] * tile_spacing[0], ny * fs[1] * tile_spacing[1]

        
    def start(self):
        #self._weights =tile_pyramid.ImagePyramid.frame_weights(self.scope.frameWrangler.currentFrame.shape[:2]).squeeze()
        
        self._scanner.genCoords()

        #metadata handling
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        self.mdh.setEntry('StartTime', time.time())
        self.mdh.setEntry('AcquisitionType', 'Tiled overview')

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(self.mdh)

        self._x0 = np.min(self._scanner.xp)  # get the upper left corner of the scan, regardless of shape/fill/start
        self._y0 = np.min(self._scanner.yp)
        
        self._pixel_size = self.mdh.getEntry('voxelsize.x')
        self._background = self.mdh.getOrDefault('Camera.ADOffset', self._background)
        
        # calculate origin independent of the camera ROI setting to store in
        # metadata for use in e.g. SupertileDatasource.DataSource.tile_coords_um
        x0_cam, y0_cam = MetaDataHandler.get_camera_physical_roi_origin(self.mdh)
            
        x0 = self._x0 + self._pixel_size*x0_cam  # offset in [um]
        y0 = self._y0 + self._pixel_size*y0_cam
        
        if self._backend_type == 'cluster':
            from PYME.Analysis import distributed_pyramid
            self.P = distributed_pyramid.DistributedImagePyramid(self._tiledir, self._base_tile_size, x0=x0, y0=y0,
                                           pixel_size=self._pixel_size)
        else:
            self.P = tile_pyramid.ImagePyramid(self._tiledir, self._base_tile_size, x0=x0, y0=y0,
                                           pixel_size=self._pixel_size)
        
        if self._save_raw:
            eventLog.register_event_handler(self.storage.event_logger)
        

        with self.scope.frameWrangler.spooling_stopped():
            if self._trigger:
                self.scope.cam.SetAcquisitionMode(self.scope.cam.MODE_SOFTWARE_TRIGGER)

            self._scanner.init_scan()
            self.scope.frameWrangler.onFrame.connect(self.on_frame, dispatch_uid=self._uuid)

        self.dtStart = datetime.datetime.now()
        if self._save_raw:
            self.storage.initialise()

        self.frame_num = 0
        
    def on_frame(self, frameData, **kwargs):
        pos = self.scope.GetPos()
        
        with self.scope.frameWrangler.spooling_paused():
            #pointScanner.PointScanner.on_frame(self, frameData, **kwargs)
            finished = self._scanner.next_pos()
        
        d = frameData.astype('f').squeeze()
        if not self._background is None:
            d = d - self._background
    
        if not self._flat is None:
            d = d * self._flat
            
        x_i = np.round(((pos['x'] - self._x0)/self._pixel_size)).astype('i')
        y_i = np.round(((pos['y'] - self._y0) / self._pixel_size)).astype('i')
        #print(pos['x'], pos['y'], x_i, y_i, d.min(), d.max())
        
        self.P.update_base_tiles_from_frame(x_i, y_i, d)

        if self._save_raw:
            self.storage.store_frame(self.frame_num, frameData)

        self.frame_num += 1
        
        if self._scanner.running:
            t = time.time()
            if t > (self._last_update_time + 1):
                self._last_update_time = t
                self.on_progress.send(self)

        else:
            self._finalise()
        
        
    def _finalise(self):
        # this does the finalisation steps that need to be done after the scan is complete.
        # It is separate from _stop(), as _stop() is called by PointScanner.on_frame() **before** the final frame
        # has been saved. We want to have the final frame saved before we do the finalisation steps.
            
        # disconnect the frame handler
        try:
            self.scope.frameWrangler.onFrame.disconnect(self.on_frame, dispatch_uid=self._uuid)
        except:
            logger.exception('Could not disconnect pointScanner tick from frameWrangler.onFrame')

        with self.scope.frameWrangler.spooling_stopped():
            if self._return_to_start:
                self._scanner.return_home()
            
            self.scope.turnAllLasersOff()
        
            if self._trigger:
                self.scope.cam.SetAcquisitionMode(self.scope.cam.MODE_CONTINUOUS)            

        
        self.on_progress.send(self)
        t_ = time.time()

        if self._save_raw:
            try:
                eventLog.remove_event_handler(self.storage.event_logger)
            except ValueError:
                pass

            self.storage.mdh.update(self.mdh)
            self.storage.finalise()
        
        logger.info('Finished tile acquisition')
        if self._backend_type == 'cluster':
            logger.info('Waiting for spoolers to empty and for base levels to be built')
        self.P.finish_base_tiles()
        
        if self._backend_type == 'cluster':
            logger.info('Base tiles built')

        logger.info('Completing pyramid (dt = %3.2f)' % (time.time()-t_))
        self.P.update_pyramid()

        if self._backend_type == 'cluster':
            from PYME.IO import clusterIO
            clusterIO.put_file(self.P.base_dir + '/metadata.json', self.P.mdh.to_JSON().encode())
        else:
            with open(os.path.join(self._tiledir, 'metadata.json'), 'w') as f:
                f.write(self.P.mdh.to_JSON())
                
        logger.info('Pyramid complete (dt = %3.2f)' % (time.time()-t_))

        self.spool_complete = True
            
        self.on_stop.send(self)
        self.on_progress.send(self)

    def stop(self):
        self._scanner.stop()
        self._finalise() # TODO - abort?? (main differences are return home, and logging messages) )

    def md(self):
        return self.mdh
    
    def _launch_viewer(self):
        # TODO - move to a UI module
        import subprocess
        import sys
        import webbrowser
        import time
        import requests
        import os

        # abs path the tile dir
        tiledir = self._tiledir
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


def Tiler(*args, **kwargs):
    logger.warning('Tiler is deprecated, please use TileAcquisition')
    return TileAcquisition(*args, **kwargs)


# class CircularTiler(Tiler, pointScanner.CircularPointScanner):
#     def __init__(self, scope, tile_dir, max_radius_um=100, tile_spacing=None, dwelltime=1, background=0, evtLog=False,
#                  trigger=False, base_tile_size=256, return_to_start=True):
#         """
#         :param return_to_start: bool
#             Flag to toggle returning home at the end of the scan. False leaves scope position as-is on scan completion.
#         """
#         if tile_spacing is None:
#             fs = np.array(scope.frameWrangler.currentFrame.shape[:2])
#             # calculate tile spacing such that there is ~30% overlap.
#             tile_spacing = (1/np.sqrt(2)) * fs * np.array(scope.GetPixelSize())
#         # take the pixel size to be the same or at least similar in both directions
#         pixel_radius = int(max_radius_um / tile_spacing.mean())
#         logger.debug('Circular tiler target radius in units of (overlapped) FOVs: %d' % pixel_radius)

#         pointScanner.CircularPointScanner.__init__(self, scope, pixel_radius,
#                                           tile_spacing, dwelltime, background, 
#                                           False, evtLog, trigger=trigger, 
#                                           stop_on_complete=True,
#                                           return_to_start=return_to_start)
        
#         self._tiledir = tile_dir
#         self._base_tile_size = base_tile_size
#         self._flat = None #currently not used
        
#         self._last_update_time = 0
        
#         self.on_stop = dispatch.Signal()
#         self.on_progress = dispatch.Signal()


class MultiwellCircularTiler(object):
    """
    Creates a circular tiler for each well at a given spacing. For now create a separate tilepyramid for each well.
    """
    def __init__(self, well_scan_radius, x_spacing, y_spacing, n_x, n_y, scope, tile_dir, tile_spacing=None,
                 dwelltime=1, background=0, evtLog=False, trigger=False, base_tile_size=256, laser_state=None):
        """
        Creates a new pyramid for each well due to performance constraints.

        Parameters
        ----------
        well_scan_radius: float
            radius to scan within each well [um]
        x_well_spacing: float
            center-to-center spacing of each well along x [um]
        y_well_spacing: float
            center-to-center spacing of each well along y [um]
        n_x: int
            number of rows along x to tile
        n_y: int
            number of columns along y to tile
        scope: PYME.Acquire.microscope
        tile_dir: str
            directory to store all pyramids
        tile_spacing: float
            distance between tile 'pixels', i.e. center-to-center distance between the individual tiles.
        dwelltime
        background
        evtLog
        trigger
        base_tile_size
        laser_state: dict
            state lasers should be in at the start of each well - lasers are blanked between wells. Should be compatible
            with PYME.Acquire.microscope.StateManager.setItems
        """

        self.well_scan_radius = well_scan_radius
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.n_x = n_x
        self.n_y = n_y

        self.scope = scope
        self.tile_dir = tile_dir

        self.set_well_positions()

        self.start_state = laser_state if laser_state is not None else {}

        # store the individual tiler settings
        self.tile_spacing = tile_spacing
        self.dwelltime = dwelltime
        self.background = background
        self.evt_log = evtLog
        self.trigger = trigger
        self.base_tile_size = base_tile_size

        # set our current well index
        self.ind = 0

    def set_well_positions(self):
        """
        Establish x,y center positions of each well to scan. Making the current microscope position the center of the
        (0, 0) well, which is also the min x, min y well.

        """
        self.curr_pos = self.scope.GetPos()

        x_wells = np.arange(0, self.n_x * self.x_spacing, self.x_spacing)
        y_wells = np.arange(0, self.n_y * self.y_spacing, self.y_spacing)

        self._x_wells = []
        self._y_wells = np.repeat(y_wells, self.n_x)
        # zig-zag with turns along x
        for xi in range(self.n_y):
            if xi % 2:
                self._x_wells.extend(x_wells[::-1])
            else:
                self._x_wells.extend(x_wells)
        self._x_wells = np.asarray(self._x_wells)

        # add the current scope position offset
        self._x_wells += self.curr_pos['x']
        self._y_wells += self.curr_pos['y']

        self.max_ind = self.n_x * self.n_y

    def start_next(self, *args, **kwargs):
        """
        Creates and starts the tiler for the next well until we're finished.

        Parameters
        ----------
        args
        kwargs:
            necessary as dispatch calls will include a signal keyword argument
        """
        try:
            self.tiler.on_stop.disconnect()
        except:
            pass

        if self.ind < self.max_ind:
            self.start_state.update({'Positioning.x': self._x_wells[self.ind],
                                     'Positioning.y': self._y_wells[self.ind]})
            self.scope.state.setItems(self.start_state,
                                      stopCamera=True)  # stop cam to make sure the next tiler gets the right center pos

            tile_dir = os.path.join(self.tile_dir, 'well_%d' % self.ind)
            self.tiler = CircularTiler(self.scope, tile_dir, self.well_scan_radius, self.tile_spacing, self.dwelltime,
                                  self.background, self.evt_log, self.trigger, self.base_tile_size, False)
            self.tiler.start()
            self.ind += 1
            self.tiler.on_stop.connect(self.start_next)


    def start(self):
        self.start_next()

    def stop(self):
        self.max_ind = 0
        try:
            self.tiler.stop()
        except AttributeError:
            pass
        
