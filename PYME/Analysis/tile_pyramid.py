import numpy as np
from PYME.IO.MetaDataHandler import get_camera_roi_origin, get_camera_physical_roi_origin, NestedClassMDHandler
import os
import glob
import collections
import time
import six
import tempfile
import logging

logger = logging.getLogger(__name__)

CacheEntry = collections.namedtuple('CacheEntry', ['data', 'saved'])

class TileCache(object):
    def __init__(self, max_size=1000):
        self._max_size = max_size
        self._cache = {}
        #self._cache_size=0
        self._cache_keys = []
        
    def _load(self, filename):
        return np.load(filename)
        
    def load(self, filename):
        try:
            return self._cache[filename].data
        except KeyError:
            data = self._load(filename)
            self._add(filename, data, saved=True)
            return data
        
    def save(self, filename, data):
        self._add(filename, data, saved=False)
        
    def _save(self, filename, data):
        dirname = os.path.split(filename)[0]
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        np.save(filename, data)
        
    def _save_entry(self, filename):
        item = self._cache[filename]
        if not item.saved:
            self._save(filename, item.data)
            self._cache[filename] = CacheEntry(data=item.data, saved=True)
            
        
    def _add(self, filename, data, saved=True):
        if filename in self._cache_keys:
            # replace existing item
            # we are going to be a bit sneaky, and remove the key before re-adding it later. This will move the key to the
            # back of the list, making it the most recently accessed (and least likely to be popped to make way for new data)
            self._cache_keys.remove(filename)
            
        if len(self._cache_keys) >= self._max_size:
            # adding item would make us too large, pop oldest entry from our cache
            fn = self._cache_keys.pop(0) #remove oldest item
            self._drop(fn)
        
        self._cache_keys.append(filename)
        self._cache[filename] = CacheEntry(data=data, saved=saved)
            
    def _drop(self, filename):
        item = self._cache.pop(filename)
        if not item.saved:
            self._save(filename, item.data)
            
    def flush(self):
        for filename, item  in list(self._cache.items()):
            if not item.saved:
                self._save(filename, item.data)
                self._cache[filename] = CacheEntry(data=item.data, saved=True)
                
    def remove(self, filename):
        if filename in self._cache_keys:
            self._cache_keys.remove(filename)
            self._cache.pop(filename)
            
        if os.path.exists(filename):
            os.remove(filename)
                
    def purge(self):
        self.flush()
        self._cache_keys = []
        self._cache = {}
        
    def exists(self, filename):
        return (filename in self._cache_keys) or os.path.exists(filename)
    
    
class PZFTileCache(TileCache):
    def _save(self, filename, data):
        from PYME.IO import PZFFormat
        dirname = os.path.split(filename)[0]
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        with open(filename, 'wb') as f:
            f.write(PZFFormat.dumps(data.astype('float32')))
    
    def _load(self, filename):
        from PYME.IO import PZFFormat
        with open(filename, 'rb') as f:
            return PZFFormat.loads(f.read())[0].squeeze()
    
class TileIO(object):
    def get_tile(self, layer, x, y):
        raise NotImplementedError
    
    def save_tile(self, layer, x, y, data):
        raise  NotImplementedError
    
    def delete_tile(self, layer, x, y):
        raise  NotImplementedError
    
    def tile_exists(self, layer, x, y):
        raise NotImplementedError
    
    def get_layer_tile_coords(self, layer):
        raise NotImplementedError
    
    def flush(self):
        pass

class NumpyTileIO(TileIO):
    def __init__(self, base_dir, suff='img'):
        self.base_dir = base_dir
        self.suff = suff + '.npy'
        
        self.pattern = os.sep.join([self.base_dir, '%d', '%03d', '%03d_%03d_' + self.suff])
        
        self._tilecache = TileCache()
        self._coords = {}
    
    def _filename(self, layer, x, y):
        return self.pattern % (layer, x, x, y)
        #return os.path.join(self.base_dir, '%d' % layer, '%03d' % x, '%03d_%03d_%s' % (x, y, self.suff))
    
    def get_tile(self, layer, x, y):
        try:
            return self._tilecache.load(self._filename(layer, x, y))
        except IOError:
            return None
        
    def save_tile(self, layer, x, y, data):
        self._check_layer_tile_coords(layer)
        self._tilecache.save(self._filename(layer, x, y), data)
        
        if not (x,y) in self._coords[layer]:
            self._coords[layer].append((x,y))
        
    def delete_tile(self, layer, x, y):
        self._check_layer_tile_coords(layer)
        self._tilecache.remove(self._filename(layer, x, y))
        self._coords[layer].remove((x, y))
        
    def tile_exists(self, layer, x, y):
        #return (x, y) in self._coords[layer]
        return self._tilecache.exists(self._filename(layer, x, y))
        
    def _check_layer_tile_coords(self, layer=0):
        if not layer in self._coords.keys():
            self._update_layer_tile_coords(layer)
    
    def _update_layer_tile_coords(self, layer=0):
        tiles = []
        for xdir in glob.glob(os.path.join(self.base_dir, '%d' % layer, '*')):
            for fn in glob.glob(os.path.join(xdir, '*_%s' % self.suff)):
                tiles.append(tuple([int(s) for s in os.path.basename(fn).split('_')[:2]]))
                
        self._coords[layer] = tiles
        
    def get_layer_tile_coords(self, layer=0):
        self._check_layer_tile_coords(layer)
        return self._coords[layer]
    
    def flush(self):
        self._tilecache.flush()

class PZFTileIO(NumpyTileIO):
    def __init__(self, base_dir, suff='img'):
        self.base_dir = base_dir
        self.suff = suff + '.pzf'

        self.pattern = os.sep.join([self.base_dir, '%d', '%03d', '%03d_%03d_' + self.suff])
    
        self._tilecache = PZFTileCache()
        self._coords = {}
        

if six.PY2:
    def blob(data):
        return buffer(data)
else:
    #Py3k
    def blob(data):
        return bytes(data)

class SqliteTileIO(TileIO):
    def __init__(self, base_dir, suff='img'):
        import sqlite3
        
        self._conn = sqlite3.connect(os.path.join(base_dir, '%s.db' % suff))
        self._cur = self._conn.cursor()

        self._known_tables =[r[0] for r in self._cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        self._coords = {}
        
        
    def get_tile(self, layer, x, y):
        from PYME.IO import PZFFormat
        table = 'layer%d' % layer
        if not table in self._known_tables:
            return None
        
        self._cur.execute('SELECT data FROM layer%d WHERE x=? AND y=?' % layer, (x, y))
        r = self._cur.fetchone()
        if r is None:
            return None
        else:
            return PZFFormat.loads(r[0])[0].squeeze()
    
    def save_tile(self, layer, x, y, data):
        from PYME.IO import PZFFormat
        table = 'layer%d' % layer
        
        if not table in self._known_tables:
            self._cur.execute('CREATE TABLE %s (y INTEGER, x INTEGER, data BLOB)' % table)
            self._cur.execute('CREATE INDEX %s ON %s (x,y)' % ('idx_' + table, table))
            self._known_tables.append(table)
        
        self._cur.execute('INSERT INTO %s VALUES (?,?,?)' % table, (x,y,blob(PZFFormat.dumps(data.astype('float32')))))
        
    def delete_tile(self, layer, x, y):
        self._cur.execute('DELETE FROM layer%d WHERE x=? AND y=?' % layer, (x, y))
        
    def tile_exists(self, layer, x, y):
        table = 'layer%d' % layer
        if not table in self._known_tables:
            return False
        
        return self._cur.execute('SELECT 1 FROM layer%d WHERE x=? AND y=?' % layer, (x, y)).fetchone() is not None

    def get_layer_tile_coords(self, layer=0):
        coords = self._cur.execute('SELECT x, y FROM layer%d' % layer).fetchall()
        #print coords
        return coords
    
    def flush(self):
        self._conn.commit()
        
    def __del__(self):
        self._cur.close()
        self._conn.close()
    

class ImagePyramid(object):
    def __init__(self, storage_directory, pyramid_tile_size=256, mdh=None, 
                 n_tiles_x = 0, n_tiles_y = 0, depth=0, x0=0, y0=0, 
                 pixel_size=1):
        
        if isinstance(storage_directory, tempfile.TemporaryDirectory):
            # If the storage directory is a temporary directory, keep a reference and cleanup the directory when we delete the pyramid
            # used to support transitory pyramids. 
            self._temp_directory = storage_directory
            storage_directory = storage_directory.name
        
        self.base_dir = storage_directory
        self.tile_size = pyramid_tile_size
        
        self.pyramid_valid = False
        
        self._mdh = NestedClassMDHandler(mdh)
        self._mdh['Pyramid.TileSize'] = self.tile_size
        
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y
        self.depth = depth
        
        self.x0 = x0
        self.y0 = y0
        self.pixel_size=pixel_size
        # TODO - should we be re-assigning these on load, not just when we create a new pyramid?
        self._mdh['Pyramid.x0'] = x0
        self._mdh['Pyramid.y0'] = y0
        self._mdh['Pyramid.PixelSize'] = pixel_size
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
        #self._tilecache = TileCache()
            
        self._imgs = PZFTileIO(base_dir=self.base_dir, suff='img')
        self._acc = PZFTileIO(base_dir=self.base_dir, suff='acc')
        self._occ = PZFTileIO(base_dir=self.base_dir, suff='occ')

        # self._imgs = SqliteTileIO(base_dir=self.base_dir, suff='img')
        # self._acc = SqliteTileIO(base_dir=self.base_dir, suff='acc')
        # self._occ = SqliteTileIO(base_dir=self.base_dir, suff='occ')
    
    def __del__(self):
        try:
            self._temp_directory.cleanup()
        except:
            pass
    
    def get_tile(self, layer, x, y):
        return self._imgs.get_tile(layer, x, y)
    
    def get_oversize_tile(self, layer, x, y, span=2):
        """
        Get an over-sized tile - allows processing on overlapping tiles

        Parameters
        ----------

        span: size of tile as a multiple of the underlying tile size

        """
        
        new_tile = np.zeros([self.tile_size * span, self.tile_size * span])
        
        for i in range(span):
            for j in range(span):
                subtile = self.get_tile(layer, x + i, y + j)
                if not subtile is None:
                    new_tile[(i * self.tile_size):((i + 1) * self.tile_size),
                    (j * self.tile_size):((j + 1) * self.tile_size)] = subtile
                    
        return new_tile
        logger.debug('Making layer %d' % (inputLevel+1))
    
    def get_layer_tile_coords(self, level):
        return self._imgs.get_layer_tile_coords(level)
    
    def _make_layer(self, inputLevel):
        from scipy import ndimage

        new_layer = inputLevel + 1
        tile_coords = self.get_layer_tile_coords(inputLevel)
        
        #print('tile_coords:', tile_coords)
        
        qsize = int(self.tile_size / 2)
        
        new_tile_coords = list(set([tuple(np.floor(np.array(tc) / 2).astype('i').tolist()) for tc in tile_coords]))
        #print('new_tile_coords:', new_tile_coords)
        
        for xc, yc in new_tile_coords:
            if not self._imgs.tile_exists(new_layer, xc, yc):
                tile = np.zeros([self.tile_size, self.tile_size])
                
                NW = self.get_tile(inputLevel, 2 * xc, 2 * yc)
                if not NW is None:
                    tile[:qsize, :qsize] = ndimage.zoom(NW, .5)
                    #print(xc, yc, 'NW')
                
                NE = self.get_tile(inputLevel, (2 * xc) + 1, (2 * yc))
                if not NE is None:
                    tile[qsize:, :qsize] = ndimage.zoom(NE, .5)
                    #print(xc, yc, 'NE')
                
                SW = self.get_tile(inputLevel, (2 * xc), (2 * yc) + 1)
                if not SW is None:
                    tile[:qsize, qsize:] = ndimage.zoom(SW, .5)
                    #print(xc, yc, 'SW')
                
                SE = self.get_tile(inputLevel, (2 * xc) + 1, (2 * yc) + 1)
                if not SE is None:
                    tile[qsize:, qsize:] = ndimage.zoom(SE, .5)
                    #print(xc, yc, 'SE')
                
                self._imgs.save_tile(new_layer, xc, yc, tile)
        
        return len(new_tile_coords)
    
    def _rebuild_base(self):
        for xc, yc in self._occ.get_layer_tile_coords(0):
            if not self._imgs.tile_exists(0, xc, yc):
                    occ = self._occ.get_tile(0, xc, yc) + 1e-9
                    sf = 1.0 / occ
                    sf[occ <= .1] = 0
                    tile_ = self._acc.get_tile(0, xc, yc) * sf

                    self._imgs.save_tile(0, xc, yc, tile_)
    
    def update_pyramid(self):
        self._rebuild_base()
        inputLevel = 0
        
        while self._make_layer(inputLevel) > 1:
            inputLevel += 1
        
        self.pyramid_valid = True
        self.depth = inputLevel
        self._imgs.flush()
    
    def _clean_tiles(self, x, y):
        level = 0
        
        while self._imgs.tile_exists(level, x, y):
            self._imgs.delete_tile(level, x, y)
            
            level += 1
            x = int(np.floor(x / 2))
            y = int(np.floor(y / 2))
            
    
    @property
    def mdh(self):
        mdh = NestedClassMDHandler(self._mdh)
        mdh['Pyramid.Depth'] = self.depth
        mdh['Pyramid.NTilesX'] = self.n_tiles_x
        mdh['Pyramid.NTilesY'] = self.n_tiles_y
        mdh['Pyramid.PixelsX'] = self.n_tiles_x * self.tile_size
        mdh['Pyramid.PixelsY'] = self.n_tiles_y * self.tile_size
        
        return mdh
    
    def add_base_tile(self, x, y, frame, weights):
        #print('add_base_tile(%d, %d)' % (x, y))

        frameSizeX, frameSizeY = frame.shape[:2]
        
        out_folder = os.path.join(self.base_dir, '0')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        tile_xs = range(int(np.floor(x / self.tile_size)), int(np.floor((x + frameSizeX) / self.tile_size) + 1))
        tile_ys = range(int(np.floor(y / self.tile_size)), int(np.floor((y + frameSizeY) / self.tile_size) + 1))
        
        #print('tile_xs: %s, tile_ys: %s' % (tile_xs, tile_ys))

        self.n_tiles_x = max(self.n_tiles_x, max(tile_xs))
        self.n_tiles_y = max(self.n_tiles_y, max(tile_ys))
        
        for tile_x in tile_xs:
            for tile_y in tile_ys:
                acc_ = self._acc.get_tile(0, tile_x, tile_y)
                occ_ = self._occ.get_tile(0, tile_x, tile_y)
                
                if (acc_ is None) or (occ_ is None):
                    acc_ = np.zeros([self.tile_size, self.tile_size])
                    occ_ = np.zeros([self.tile_size, self.tile_size])
                
                xs, xe = max(tile_x * self.tile_size - x, 0), min((tile_x + 1) * self.tile_size - x, frameSizeX)
                xst = max(x - tile_x * self.tile_size, 0)
                xet = min(xst + (xe - xs),
                          self.tile_size) #min(frameSizeX - (tile_x + 1) * self.tile_size - x, 0) #FIXME
                
                ys, ye = max((tile_y * self.tile_size) - y, 0), min(((tile_y + 1) * self.tile_size) - y,
                                                                  frameSizeY)
                
                yst = max(y - tile_y * self.tile_size, 0)
                yet = min(yst + (ye - ys), self.tile_size) #min(frameSizeY - (tile_y + 1) * self.tile_size - y,0) #FIXME
                
                #print(tile_x, tile_y)
                #print('tile[%d:%d, %d:%d] = frame[%d:%d, %d:%d]' % (xst, xet, yst, yet, xs, xe, ys, ye))
                acc_[xst:xet, yst:yet] += frame[xs:xe, ys:ye]
                occ_[xst:xet, yst:yet] += weights[xs:xe, ys:ye]

                self._acc.save_tile(0, tile_x, tile_y, acc_)
                self._occ.save_tile(0, tile_x, tile_y, occ_)
                
                self._clean_tiles(tile_x, tile_y)
        
        self.pyramid_valid = False


def get_position_from_events(events, mdh):
    """Use acquisition events to create a mapping between frame number and
    stage position
    """
    from PYME.Analysis import piecewiseMapping
    x0 = mdh.getOrDefault('Positioning.x', 0)
    y0 = mdh.getOrDefault('Positioning.y', 0)
    
    xm = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], x0, b'ScannerXPos', 0)
    ym = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], y0, b'ScannerYPos', 0)
    
    return xm, ym


def tile_pyramid(out_folder, ds, xm, ym, mdh, split=False, skipMoveFrames=False, shiftfield=None,
                 mixmatrix=[[1., 0.], [0., 1.]],
                 correlate=False, dark=None, flat=None, pyramid_tile_size=256):
    """Create a tile pyramid from which an ImagePyramid can be created

    Parameters
    ----------
    out_folder : str
        directory to save pyramid tiles(/directories)
    ds : PYME.IO.DataSources.BaseDataSource, np.ndarray
        array-like image
    xm : np.ndarray or PYME.Analysis.piecewiseMapping.piecewiseMap
        x positions of frames in ds
    ym : np.ndarray or PYME.Analysis.piecewiseMapping.piecewiseMap
        y positions of frames in ds
    mdh : PYME.IO.MetaDataHandler.MDataHandlerBase
        metadata for ds
    split : bool, optional
        whether this is a splitter datasource and should be treated like one,
        by default False
    skipMoveFrames : bool, optional
        flag to drop frames which are the first frame acquired at a given
        position, by default False
    shiftfield : [type], optional
        required for splitter data, see PYME.Acquire.Hardware.splitter, by 
        default None
    mixmatrix : list, optional
        for splitter data, see PYME.Acquire.Hardware.splitter, by 
        default [[1., 0.], [0., 1.]]
    correlate : bool, optional
        whether to add a 300 pixel padding to the edges, by default False
    dark : ndarray, float, optional
        (appropriately-cropped or scalar) dark frame (analog-digital offset)
        calibration to subtract when adding frames to the pyramid, by default
        None, in which case Camera.ADOffset from metadata will be used, if 
        available
    flat : ndarray, optional
        (appropriately-cropped or scalar) flatfield calibration to apply to 
        frames when adding them to the pyramid, by default None
    pyramid_tile_size : int, optional
        base tile size, by default 256 pixels

    Returns
    -------
    ImagePyramid
        coalesced/averaged/etc multilevel ImagePyramid instance
    """
    frameSizeX, frameSizeY, numFrames = ds.shape[:3]
    
    if split:
        from PYME.Acquire.Hardware import splitter
        frameSizeY /= 2
        nchans = 2
        unmux = splitter.Unmixer(shiftfield, mdh.voxelsize_nm.x)
    else:
        nchans = 1
    
    #x & y positions of each frame
    xps = xm(np.arange(numFrames)) if not isinstance(xm, np.ndarray) else xm
    yps = ym(np.arange(numFrames)) if not isinstance(ym, np.ndarray) else ym

    if mdh.getOrDefault('CameraOrientation.FlipX', False):
        xps = -xps
    
    if mdh.getOrDefault('CameraOrientation.FlipY', False):
        yps = -yps

    rotate_cam = mdh.getOrDefault('CameraOrientation.Rotate', False)

    #give some room at the edges
    bufSize = 0
    if correlate:
        bufSize = 300
    

    # make our x0, y0 independent of the camera ROI setting
    x0_cam, y0_cam = get_camera_physical_roi_origin(mdh)
    x0 = xps.min() + mdh.voxelsize_nm.x / 1e3 * x0_cam
    y0 = yps.min() + mdh.voxelsize_nm.y / 1e3 * y0_cam
    xps -= x0
    yps -= y0

    #convert to pixels
    xdp = (bufSize + (xps / (mdh.getEntry('voxelsize.x'))).round()).astype('i')
    ydp = (bufSize + (yps / (mdh.getEntry('voxelsize.y'))).round()).astype('i')
    
    #calculate a weighting matrix (to allow feathering at the edges - TODO)
    weights = np.ones((frameSizeX, frameSizeY, nchans))
    #weights[:, :10, :] = 0 #avoid splitter edge artefacts
    #weights[:, -10:, :] = 0
    
    #print weights[:20, :].shape
    edgeRamp = min(100, int(.25 * ds.shape[0]))
    weights[:edgeRamp, :, :] *= np.linspace(0, 1, edgeRamp)[:, None, None]
    weights[-edgeRamp:, :, :] *= np.linspace(1, 0, edgeRamp)[:, None, None]
    weights[:, :edgeRamp, :] *= np.linspace(0, 1, edgeRamp)[None, :, None]
    weights[:, -edgeRamp:, :] *= np.linspace(1, 0, edgeRamp)[None, :, None]
    
    roi_x0, roi_y0 = get_camera_roi_origin(mdh)
    
    ROIX1 = roi_x0 + 1
    ROIY1 = roi_y0 + 1
    
    ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
    ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')
    
    if dark is None:
        dark = float(mdh.getOrDefault('Camera.ADOffset', 0))

    P = ImagePyramid(out_folder, pyramid_tile_size, x0=x0, y0=y0, 
                     pixel_size=mdh.getEntry('voxelsize.x'))

    logger.debug('Adding base tiles ...')
    
    t1 = time.time()
    for i in range(int(mdh.getEntry('Protocol.DataStartsAt')), numFrames):
        if xdp[i - 1] == xdp[i] or not skipMoveFrames:
            x_i = xdp[i]
            y_i = ydp[i]
            d = ds[:, :, i].astype('f') - dark
            if not flat is None:
                d = d * flat
            
            if split:
                d = np.concatenate(unmux.Unmix(d, mixmatrix, dark, [ROIX1, ROIY1, ROIX2, ROIY2]), 2)

            d_weighted = weights * d


            # orient frame - TODO - check if we need to flip x and y?!
            if rotate_cam:
                #print('adding base tile from frame %d [transposed]' % i)
                P.add_base_tile(x_i, y_i, d_weighted.T.squeeze(), weights.T.squeeze())
            else:
                #print('adding base tile from frame %d' % i)
                P.add_base_tile(x_i, y_i, d_weighted.squeeze(), weights.squeeze())
                
    
    t2 = time.time()
    logger.debug('Added base tiles in %fs' % (t2 - t1))
    #P._occ.flush()
    logger.debug(time.time() - t2)
    logger.debug('Updating pyramid ...')
    P.update_pyramid()
    logger.debug(time.time() - t2)
    logger.debug('Done')

    with open(os.path.join(P.base_dir, 'metadata.json'), 'w') as f:
        f.write(P.mdh.to_JSON())
    
    return P

def create_pyramid_from_dataset(filename, outdir, tile_size=128, **kwargs):
    from PYME.IO import image
    dataset = image.ImageStack(filename=filename)
    
    xm, ym = get_position_from_events(dataset.events, dataset.mdh)
    
    #print(xm(np.arange(dataset.data.shape[2])))
    #print(ym(np.arange(dataset.data.shape[2])))
    
    p = tile_pyramid(outdir, dataset.data, xm, ym, dataset.mdh, pyramid_tile_size=tile_size)
    
    return p
        
        
        
if __name__ == '__main__':
    import sys
    from PYME.util import mProfile
    #mProfile.profileOn(['tile_pyramid.py',])
    input_stack, output_dir = sys.argv[1:]
    import time
    t1 = time.time()
    create_pyramid_from_dataset(input_stack, output_dir)
    logger.debug(time.time() - t1)
    #mProfile.report()
    
    
    
    
