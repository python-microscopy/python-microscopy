from . import pointScanner
from PYME.Analysis import tile_pyramid
import numpy as np
import time
from PYME.IO import MetaDataHandler
import os
from PYME.contrib import dispatch
import logging
logger = logging.getLogger(__name__)

class Tiler(pointScanner.PointScanner):
    def __init__(self, scope, tile_dir, n_tiles = 10, tile_spacing=None, dwelltime = 1, background=0, evtLog=False,
                 trigger=False, base_tile_size=256, return_to_start=True, backend='file'):
        """
        :param return_to_start: bool
            Flag to toggle returning home at the end of the scan. False leaves scope position as-is on scan completion.
        """
        
        if tile_spacing is None:
            fs = np.array(scope.frameWrangler.currentFrame.shape[:2])
            #calculate tile spacing such that there is 20% overlap.
            tile_spacing = 0.8*fs*np.array(scope.GetPixelSize())
            
        pointScanner.PointScanner.__init__(self, scope=scope, pixels=n_tiles, pixelsize=tile_spacing,
                                           dwelltime=dwelltime, background=background, avg=False, evtLog=evtLog,
                                           trigger=trigger, stop_on_complete=True, return_to_start=return_to_start)
        
        self._tiledir = tile_dir
        self._base_tile_size = base_tile_size
        self._flat = None #currently not used
        
        self._last_update_time = 0
        
        self._backend = backend
        
        self.on_stop = dispatch.Signal()
        self.progress = dispatch.Signal()
        
    
        
    def start(self):
        #self._weights =tile_pyramid.ImagePyramid.frame_weights(self.scope.frameWrangler.currentFrame.shape[:2]).squeeze()
        
        self.genCoords()

        #metadata handling
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        self.mdh.setEntry('StartTime', time.time())
        self.mdh.setEntry('AcquisitionType', 'Tiled overview')

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(self.mdh)

        self._x0 = np.min(self.xp)  # get the upper left corner of the scan, regardless of shape/fill/start
        self._y0 = np.min(self.yp)
        
        self._pixel_size = self.mdh.getEntry('voxelsize.x')
        self.background = self.mdh.getOrDefault('Camera.ADOffset', self.background)
        
        # calculate origin independent of the camera ROI setting to store in
        # metadata for use in e.g. SupertileDatasource.DataSource.tile_coords_um
        x0_cam, y0_cam = MetaDataHandler.get_camera_physical_roi_origin(self.mdh)
            
        x0 = self._x0 + self._pixel_size*x0_cam  # offset in [um]
        y0 = self._y0 + self._pixel_size*y0_cam
        
        if self._backend == 'cluster':
            from PYME.Analysis import distributed_pyramid
            self.P = distributed_pyramid.DistributedImagePyramid(self._tiledir, self._base_tile_size, x0=x0, y0=y0,
                                           pixel_size=self._pixel_size)
        else:
            self.P = tile_pyramid.ImagePyramid(self._tiledir, self._base_tile_size, x0=x0, y0=y0,
                                           pixel_size=self._pixel_size)
        
        pointScanner.PointScanner.start(self)
        
    def tick(self, frameData, **kwargs):
        pos = self.scope.GetPos()
        pointScanner.PointScanner.tick(self, frameData, **kwargs)
        
        d = frameData.astype('f').squeeze()
        if not self.background is None:
            d = d - self.background
    
        if not self._flat is None:
            d = d * self._flat
            
        x_i = np.round(((pos['x'] - self._x0)/self._pixel_size)).astype('i')
        y_i = np.round(((pos['y'] - self._y0) / self._pixel_size)).astype('i')
        #print(pos['x'], pos['y'], x_i, y_i, d.min(), d.max())
        
        self.P.update_base_tiles_from_frame(x_i, y_i, d)
        
        t = time.time()
        if t > (self._last_update_time + 1):
            self._last_update_time = t
            self.progress.send(self)
        
    def _stop(self):
        pointScanner.PointScanner._stop(self, send_stop=False)
        self.progress.send(self)
        t_ = time.time()
        
        logger.info('Finished tile acquisition')
        if self._backend == 'cluster':
            logger.info('Waiting for spoolers to empty and for base levels to be built')
        self.P.finish_base_tiles()
        
        if self._backend == 'cluster':
            logger.info('Base tiles built')

        logger.info('Completing pyramid (dt = %3.2f)' % (time.time()-t_))
        self.P.update_pyramid()

        if self._backend == 'cluster':
            from PYME.IO import clusterIO
            clusterIO.put_file(self.P.base_dir + '/metadata.json', self.P.mdh.to_JSON().encode())
        else:
            with open(os.path.join(self._tiledir, 'metadata.json'), 'w') as f:
                f.write(self.P.mdh.to_JSON())
                
        logger.info('Pyramid complete (dt = %3.2f)' % (time.time()-t_))
            
        self.on_stop.send(self)
        self.progress.send(self)

class CircularTiler(Tiler, pointScanner.CircularPointScanner):
    def __init__(self, scope, tile_dir, max_radius_um=100, tile_spacing=None, dwelltime=1, background=0, evtLog=False,
                 trigger=False, base_tile_size=256, return_to_start=True):
        """
        :param return_to_start: bool
            Flag to toggle returning home at the end of the scan. False leaves scope position as-is on scan completion.
        """
        if tile_spacing is None:
            fs = np.array(scope.frameWrangler.currentFrame.shape[:2])
            # calculate tile spacing such that there is ~30% overlap.
            tile_spacing = (1/np.sqrt(2)) * fs * np.array(scope.GetPixelSize())
        # take the pixel size to be the same or at least similar in both directions
        pixel_radius = int(max_radius_um / tile_spacing.mean())
        logger.debug('Circular tiler target radius in units of (overlapped) FOVs: %d' % pixel_radius)

        pointScanner.CircularPointScanner.__init__(self, scope, pixel_radius,
                                          tile_spacing, dwelltime, background, 
                                          False, evtLog, trigger=trigger, 
                                          stop_on_complete=True,
                                          return_to_start=return_to_start)
        
        self._tiledir = tile_dir
        self._base_tile_size = base_tile_size
        self._flat = None #currently not used
        
        self._last_update_time = 0
        
        self.on_stop = dispatch.Signal()
        self.progress = dispatch.Signal()


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
        
