import numpy as np
from PYME.IO.MetaDataHandler import get_camera_roi_origin, NestedClassMDHandler
import os
import glob

class ImagePyramid(object):
    def __init__(self, storage_directory, pyramid_tile_size=256, mdh=None, n_tiles_x = 0, n_tiles_y = 0, depth=0, x0=0,
                 y0=0, pixel_size=1, roi_width=None, roi_height=None):
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
        # fixme - what is this doing? Why are we overriding metadata no matter what?
        self._mdh['Pyramid.x0'] = x0
        self._mdh['Pyramid.y0'] = y0
        self._mdh['Pyramid.PixelSize'] = pixel_size
        # this metadata is gross, but allows us to account for camera orientation and swap width / height if needed.
        self._mdh['Pyramid.ROIWidth'] = roi_width if roi_width is not None else mdh.getOrDefault('Camera.ROIWidth', 0)
        self._mdh['Pyramid.ROIHeight'] = roi_width if roi_width is not None else mdh.getOrDefault('Camera.ROIHeight', 0)
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def get_tile(self, layer, x, y):
        fname = os.path.join(self.base_dir, '%d' % layer, '%03d' % x, '%03d_%03d_img.npy' % (x, y))
        try:
            return np.load(fname)
        except IOError:
            return None
    
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
        # print('Making layer %d' % (inputLevel+1))
    
    def get_layer_tile_coords(self, level):
        base_tile_dir = os.path.join(self.base_dir, '%d' % level)
    
        x_dirs = glob.glob(os.path.join(base_tile_dir, '*'))
        base_tile_names = []
    
        for x_dir in x_dirs:
            base_tile_names += glob.glob(os.path.join(x_dir, '*img.npy'))
    
        tile_coords = [np.array([int(s) for s in os.path.split(fn)[-1].split('_')[:2]]) for fn in base_tile_names]
        
        return tile_coords
    
    def _make_layer(self, inputLevel):
        from scipy import ndimage
        
        out_dir = os.path.join(self.base_dir, '%d' % (inputLevel + 1))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        tile_coords = self.get_layer_tile_coords(inputLevel)
        
        #print('tile_coords:', tile_coords)
        
        qsize = int(self.tile_size / 2)
        
        new_tile_coords = list(set([tuple(np.floor(tc / 2).astype('i').tolist()) for tc in tile_coords]))
        #print('new_tile_coords:', new_tile_coords)
        
        for xc, yc in new_tile_coords:
            x_out_dir = os.path.join(out_dir, '%03d' % xc)
            if not os.path.exists(x_out_dir):
                os.makedirs(x_out_dir)
            
            out_filename = os.path.join(x_out_dir, '%03d_%03d_img.npy' % (xc, yc))
            
            if not os.path.exists(out_filename):
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
                
                np.save(out_filename, tile)
        
        return len(new_tile_coords)
    
    def _rebuild_base(self):
        for xdir in glob.glob(os.path.join(self.base_dir, '0', '*')):
            for fn in glob.glob(os.path.join(xdir, '*_occ.npy')):
                out_fn = fn[:-7] + 'img.npy'
                
                if not os.path.exists(out_fn):
                    occ = np.load(fn)
                    sf = 1.0 / occ
                    sf[occ <= .1] = 0
                    tile_ = np.load(fn[:-7] + 'acc.npy') * sf
                    
                    np.save(out_fn, tile_)
    
    def update_pyramid(self):
        self._rebuild_base()
        inputLevel = 0
        
        while self._make_layer(inputLevel) > 1:
            inputLevel += 1
        
        self.pyramid_valid = True
        self.depth = inputLevel
    
    def _clean_tiles(self, x, y):
        level = 0
        
        tn = os.path.join(self.base_dir, '%d' % level, '%03d' % x, '%03d_%03d_img.npy' % (x, y))
        
        while os.path.exists(tn):
            os.remove(tn)
            
            level += 1
            x = int(np.floor(x / 2))
            y = int(np.floor(y / 2))
            
            tn = os.path.join(self.base_dir, '%d' % level, '%03d' % x, '%03d_%03d_img.npy' % (x, y))
            
    
    @property
    def mdh(self):
        mdh = NestedClassMDHandler(self._mdh)
        mdh['Pyramid.Depth'] = self.depth
        mdh['Pyramid.NTilesX'] = self.n_tiles_x
        mdh['Pyramid.NTilesY'] = self.n_tiles_y
        mdh['Pyramid.PixelsX'] = self.n_tiles_x * self.tile_size
        mdh['Pyramid.PixelsY'] = self.n_tiles_x * self.tile_size
        
        return mdh
    
    def add_base_tile(self, x, y, frame, weights):
        # print('add_base_tile(%d, %d)' % (x, y))

        frameSizeX, frameSizeY = frame.shape[:2]
        
        out_folder = os.path.join(self.base_dir, '0')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        tile_xs = range(int(np.floor(x / self.tile_size)), int(np.floor((x + frameSizeX) / self.tile_size) + 1))
        tile_ys = range(int(np.floor(y / self.tile_size)), int(np.floor((y + frameSizeY) / self.tile_size) + 1))
        
        # print('tile_xs: %s, tile_ys: %s' % (tile_xs, tile_ys))

        self.n_tiles_x = max(self.n_tiles_x, max(tile_xs))
        self.n_tiles_y = max(self.n_tiles_y, max(tile_ys))
        
        for tile_x in tile_xs:
            x_out_dir = os.path.join(out_folder, '%03d' % tile_x)
            if not os.path.exists(x_out_dir):
                os.makedirs(x_out_dir)
            
            for tile_y in tile_ys:
                tile_filename = os.path.join(x_out_dir, '%03d_%03d_acc.npy' % (tile_x, tile_y))
                occ_filename = os.path.join(x_out_dir, '%03d_%03d_occ.npy' % (tile_x, tile_y))
                
                try:
                    tile_ = np.load(tile_filename)
                    occ_ = np.load(occ_filename)
                except IOError:
                    tile_ = np.zeros([self.tile_size, self.tile_size])
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
                tile_[xst:xet, yst:yet] += frame[xs:xe, ys:ye]
                occ_[xst:xet, yst:yet] += weights[xs:xe, ys:ye]
                
                np.save(tile_filename, tile_)
                np.save(occ_filename, occ_)
                
                self._clean_tiles(tile_x, tile_y)
        
        self.pyramid_valid = False


def get_position_from_events(events, mdh):
    from PYME.Analysis import piecewiseMapping
    x0 = mdh.getOrDefault('Positioning.x', 0)
    y0 = mdh.getOrDefault('Positioning.y', 0)
    
    xm = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], x0, b'ScannerXPos', 0)
    ym = piecewiseMapping.GeneratePMFromEventList(events, mdh, mdh['StartTime'], y0, b'ScannerYPos', 0)
    
    return xm, ym


def tile_pyramid(out_folder, ds, xm, ym, mdh, split=False, skipMoveFrames=False, shiftfield=None,
                 mixmatrix=[[1., 0.], [0., 1.]],
                 correlate=False, dark=None, flat=None, pyramid_tile_size=256):
    frameSizeX, frameSizeY, numFrames = ds.shape[:3]
    
    if split:
        from PYME.Acquire.Hardware import splitter
        frameSizeY /= 2
        nchans = 2
        unmux = splitter.Unmixer(shiftfield, 1e3 * mdh.getEntry('voxelsize.x'))
    else:
        nchans = 1
    
    # x & y positions of each frame
    xps = xm(np.arange(numFrames))
    yps = ym(np.arange(numFrames))
    # no need to rotate positions, since xps and yps are the same
    if mdh.getOrDefault('CameraOrientation.FlipX', False):
        xps = -xps
    
    if mdh.getOrDefault('CameraOrientation.FlipY', False):
        yps = -yps

    rotate_cam = mdh.getOrDefault('CameraOrientation.Rotate', False)

    #give some room at the edges
    bufSize = 0
    if correlate:
        bufSize = 300
    

    x0 = xps.min()
    y0 = yps.min()
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

    roi_width = mdh.getEntry('Camera.ROIWidth')
    roi_height = mdh.getEntry('Camera.ROIHeight')
    # if rotate_cam:
    #     roi_width, roi_height = roi_height, roi_width
    ROIX2 = ROIX1 + roi_width
    ROIY2 = ROIY1 + roi_height
    
    if dark is None:
        offset = float(mdh.getEntry('Camera.ADOffset'))
    else:
        offset = 0.

    P = ImagePyramid(out_folder, pyramid_tile_size, x0=x0, y0=y0, pixel_size=mdh.getEntry('voxelsize.x'),
                     roi_width=roi_width, roi_height=roi_height)
    
    for i in range(int(mdh.getEntry('Protocol.DataStartsAt')), numFrames):
        if xdp[i - 1] == xdp[i] or not skipMoveFrames:
            x_i = xdp[i]
            y_i = ydp[i]
            d = ds[:, :, i].astype('f')
            if not dark is None:
                d = d - dark
            if not flat is None:
                d = d * flat
            
            if split:
                d = np.concatenate(unmux.Unmix(d, mixmatrix, offset, [ROIX1, ROIY1, ROIX2, ROIY2]), 2)

            d_weighted = weights * d


            # orient frame - TODO - check if we need to flip x and y?!
            if rotate_cam:
                print('adding base tile from frame %d [transposed]' % i)
                P.add_base_tile(x_i, y_i, d_weighted.T.squeeze(), weights.T.squeeze())
            else:
                print('adding base tile from frame %d' % i)
                P.add_base_tile(x_i, y_i, d_weighted.squeeze(), weights.squeeze())
    
    P.update_pyramid()
    
    return P

def create_pyramid_from_dataset(filename, outdir, tile_size=128, **kwargs):
    from PYME.IO import image
    dataset = image.ImageStack(filename=filename)
    
    xm, ym = get_position_from_events(dataset.events, dataset.mdh)
    
    print(xm(np.arange(dataset.data.shape[2])))
    print(ym(np.arange(dataset.data.shape[2])))
    
    p = tile_pyramid(outdir, dataset.data, xm, ym, dataset.mdh, pyramid_tile_size=tile_size)

    with open(os.path.join(outdir, 'metadata.json'), 'w') as f:
        f.write(p.mdh.to_JSON())
        
        
        
if __name__ == '__main__':
    import sys
    input_stack, output_dir = sys.argv[1:]
    
    create_pyramid_from_dataset(input_stack, output_dir)
    
    
    
    
    