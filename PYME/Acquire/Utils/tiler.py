from . import pointScanner
from PYME.Analysis import tile_pyramid
import numpy as np
import time
from PYME.IO import MetaDataHandler
import os
import dispatch
import logging
logger = logging.getLogger(__name__)

class Tiler(pointScanner.PointScanner):
    def __init__(self, scope, tile_dir, n_tiles = 10, tile_spacing=None, dwelltime = 1, background=0, evtLog=False,
                 trigger=False, base_tile_size=256, return_to_start=True):
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
        
        self.on_stop = dispatch.Signal()
        self.progress = dispatch.Signal()
        
    
    def _gen_weights(self):
        sh = self.scope.frameWrangler.currentFrame.shape[:2]
        self._weights = np.ones(sh)

        edgeRamp = min(100, int(.25 * sh[0]))
        self._weights[:edgeRamp, :] *= np.linspace(0, 1, edgeRamp)[:, None]
        self._weights[-edgeRamp:, :,] *= np.linspace(1, 0, edgeRamp)[:, None]
        self._weights[:, :edgeRamp] *= np.linspace(0, 1, edgeRamp)[None, :]
        self._weights[:, -edgeRamp:] *= np.linspace(1, 0, edgeRamp)[None, :]
        
    def start(self):
        self._gen_weights()
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
        
        # make our x0, y0 independent of the camera ROI setting
        x0_cam, y0_cam = MetaDataHandler.get_camera_physical_roi_origin(self.mdh)
            
        x0 = self._x0 + self._pixel_size*x0_cam
        y0 = self._y0 + self._pixel_size*y0_cam
        
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
        print(pos['x'], pos['y'], x_i, y_i, d.min(), d.max())
        
        self.P.add_base_tile(x_i, y_i, self._weights*d, self._weights)
        
        t = time.time()
        if t > (self._last_update_time + 1):
            self._last_update_time = t
            self.progress.send(self)
        
    def _stop(self):
        pointScanner.PointScanner._stop(self)
        
        self.P.update_pyramid()

        with open(os.path.join(self._tiledir, 'metadata.json'), 'w') as f:
            f.write(self.P.mdh.to_JSON())
            
        self.on_stop.send(self)
        self.progress.send(self)

class CircularTiler(Tiler):
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
        self.pixel_radius = int(max_radius_um / tile_spacing.mean())
        logger.debug('Circular tiler target radius in units of ~30 percent overlapped FOVs: %d' % self.pixel_radius)
        
        Tiler.__init__(self, scope, tile_dir, n_tiles=self.pixel_radius, tile_spacing=tile_spacing, dwelltime=dwelltime,
                       background=background, evtLog=evtLog, trigger=trigger, base_tile_size=base_tile_size,
                       return_to_start=return_to_start)

    def genCoords(self):
        """
        Generate coordinates for square ROIs evenly distributed within a circle. Order them first by radius, and then
        by increasing theta such that the initial position is scanned first, and then subsequent points are scanned in
        an ~optimal order.
        """
        self.currPos = self.scope.GetPos()
        logger.debug('Current positions: %s' % (self.currPos,))
    
        r, t = [0], [np.array([0])]
        for r_ring in self.pixelsize[0] * np.arange(1, self.pixel_radius + 1):  # 0th ring is (0, 0)
            # keep the rings spaced by pixel size and hope the overlap is enough
            # 2 pi / (2 pi r / pixsize) = pixsize/r
            thetas = np.arange(0, 2 * np.pi, self.pixelsize[0] / r_ring)
            r.extend(r_ring * np.ones_like(thetas))
            t.append(thetas)
    
        # convert to cartesian and add currPos offset
        r = np.asarray(r)
        t = np.concatenate(t)
        self.xp = r * np.cos(t) + self.currPos['x']
        self.yp = r * np.sin(t) + self.currPos['y']
    
        self.nx = len(self.xp)
        self.ny = len(self.yp)
        self.imsize = self.nx

    def _position_for_index(self, callN):
        ind = callN % self.nx
        return self.xp[ind], self.yp[ind]

        
