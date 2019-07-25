from . import pointScanner
from PYME.Analysis import tile_pyramid
import numpy as np
import time
from PYME.IO import MetaDataHandler
import os
import dispatch

class Tiler(pointScanner.PointScanner):
    def __init__(self, scope, tile_dir, n_tiles = 10, tile_spacing=None, dwelltime = 1, background=0, evtLog=False,
                 trigger=False, base_tile_size=256):
        
        if tile_spacing is None:
            fs = np.array(scope.frameWrangler.currentFrame.shape[:2])
            #calculate tile spacing such that there is 20% overlap.
            tile_spacing = 0.8*fs*np.array(scope.GetPixelSize())
            
        pointScanner.PointScanner.__init__(self, scope=scope, pixels=n_tiles, pixelsize=tile_spacing,  dwelltime=dwelltime,
                                           background =background, avg=False, evtLog=evtLog, trigger=trigger, stop_on_complete=True)
        
        self._tiledir = tile_dir
        self._base_tile_size = base_tile_size
        self._flat = None #currently not used
        
        self.on_stop = dispatch.Signal()
        
    
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

        self._x0 = self.xp[0]
        self._y0 = self.yp[0]
        
        self._pixel_size = self.mdh.getEntry('voxelsize.x')
        self.background = self.mdh.getOrDefault('Camera.ADOffset', self.background)
        
        self.P = tile_pyramid.ImagePyramid(self._tiledir, self._base_tile_size, x0=self._x0, y0=self._y0,
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
        
    def _stop(self):
        pointScanner.PointScanner._stop(self)
        
        self.P.update_pyramid()

        with open(os.path.join(self._tiledir, 'metadata.json'), 'w') as f:
            f.write(self.P.mdh.to_JSON())
            
        self.on_stop.send(self)
