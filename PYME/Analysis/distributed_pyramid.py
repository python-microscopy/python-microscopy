import numpy as np
from PYME.IO.MetaDataHandler import get_camera_roi_origin, get_camera_physical_roi_origin, load_json, NestedClassMDHandler


import threading
import queue
import os
import glob
import collections
import time
import six
import tempfile
import socket
import requests
import logging
import json
import queue
import logging
from PYME.IO import clusterIO
from .tile_pyramid import TILEIO_EXT, ImagePyramid, get_position_from_events, PZFTileIO


logger = logging.getLogger(__name__)


def server_for_chunk(self, x, y, z=0, chunk_shape=[8,8,1], nr_servers=1):
    """
    Returns the server responsible for the chunk of tiles at given (x, y, z).
    """
    server_id = np.floor(x/chunk_shape[0])
    server_id = server_id + np.floor(y/chunk_shape[1]) * nr_servers
    server_id = server_id + np.floor(z/chunk_shape[2]) % nr_servers
    server_id = int(server_id)
    return server_id


def update_partial_pyramid(pyramid, sleep_time=1):
    while not pyramid.all_tiles_received:
        time.sleep(sleep_time)
        while not pyramid.update_queue.empty():
            data = pyramid.update_queue.get()
            pyramid.update_base_tile_from_request_data(data)


class PartialPyramid(ImagePyramid):

    """
    Subclass of ImagePyramid which supports distribution of pyramid files over a PYME cluster.
    Implementation from the microscope side.
    """
    def __init__(
        self, storage_directory, pyramid_tile_size=256, mdh=None, 
        n_tiles_x=0, n_tiles_y=0, depth=0, x0=0, y0=0, 
        pixel_size=1, backend=PZFTileIO, chunk_shape=[8,8,1], nr_servers=1, 
        server_idx=0,
    ):
        super().__init__(
            storage_directory, pyramid_tile_size=pyramid_tile_size, mdh=mdh,
            n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, depth=depth, x0=x0, y0=y0, 
            pixel_size=pixel_size, backend=backend
        )
        self.chunk_shape = chunk_shape
        self.server_idx = server_idx
        self.update_queue = queue.SimpleQueue()
        self.update_thread = threading.Thread(target=update_partial_pyramid, args=(self,), daemon=True)
        self.all_tiles_received = False

    @classmethod
    def build(self, filepath, part_dict):
        part_params = json.loads(part_dict.decode())
        return PartialPyramid(
            filepath,
            pyramid_tile_size=part_params["pyramid_tile_size"],
            # mdh=part_dict["mdh"],
            n_tiles_x=part_params["n_tiles_x"],
            n_tiles_y=part_params["n_tiles_y"],
            depth=part_params["depth"],
            x0=part_params["x0"],
            y0=part_params["y0"],
            pixel_size=part_params["pixel_size"],
            backend=TILEIO_EXT[part_params["backend"]],
            chunk_shape=part_params["chunk_shape"],
            nr_servers=part_params["nr_servers"],
            server_idx=part_params["server_idx"],
        )

    def update_base_tile_from_request_data(self, data):
        data_dict = json.loads(data.decode())
        frame_slice = np.asarray(data_dict["frame_data"])
        frame_slice = frame_slice.reshape(data_dict["frame_shape"])
        weights_slice = np.asarray(data_dict["weights_data"])
        weights_slice = weights_slice.reshape(data_dict["weights_shape"])
        coords = data_dict["coords"]
        self.update_base_tile_from_slices(0, coords, frame_slice, weights_slice)

    def update_base_tile_from_slices(self, layer, coords, frame_slice, weights_slice):
        [tile_x, tile_y, xst, xet, yst, yet] = coords
        acc_ = self._acc.get_tile(layer, tile_x, tile_y)
        occ_ = self._occ.get_tile(layer, tile_x, tile_y)

        if (acc_ is None) or (occ_ is None):
            acc_ = np.zeros([self.tile_size, self.tile_size])
            occ_ = np.zeros([self.tile_size, self.tile_size])

        acc_[xst:xet, yst:yet] += frame_slice
        occ_[xst:xet, yst:yet] += weights_slice
        self._acc.save_tile(layer, tile_x, tile_y, acc_)
        self._occ.save_tile(layer, tile_x, tile_y, occ_)

        self._clean_tiles(tile_x, tile_y)



class DistributedImagePyramid(ImagePyramid):

    """
    Subclass of ImagePyramid which supports distribution of pyramid files over a PYME cluster.
    Implementation from the microscope side.
    """
    def __init__(
        self, storage_directory, pyramid_tile_size=256, mdh=None, 
        n_tiles_x=0, n_tiles_y=0, depth=0, x0=0, y0=0, 
        pixel_size=1, backend=PZFTileIO, n_servers=1, chunk_shape=[8,8,1], timeout=10, repeats=3
    ):
        super().__init__(
            storage_directory, pyramid_tile_size=pyramid_tile_size, mdh=mdh,
            n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, depth=depth, x0=x0, y0=y0, 
            pixel_size=pixel_size, backend=backend
        )
        self.logger = logging.getLogger(__name__)
        self.chunk_shape = chunk_shape
        self.timeout = timeout
        self.repeats = repeats
        self.servers = [((k), (v)) for k, v in clusterIO.get_ns().get_advertised_services()]
        self.sessions = [requests.Session() for _, _ in self.servers]
        for server_idx in range(len(self.servers)):
            self.create_remote_part(server_idx, backend)

    def part_dict(self, server_idx, backend):
        part_params = {
            "pyramid_tile_size": self.tile_size,
            # "mdh": self.mdh, # turn to json if possible
            "n_tiles_x": self.n_tiles_x,
            "n_tiles_y": self.n_tiles_y,
            "depth": self.depth,
            "x0": self.x0,
            "y0": self.y0,
            "pixel_size": self.pixel_size,
            "backend": [key for key in TILEIO_EXT if TILEIO_EXT[key] == backend][0],
            "chunk_shape": self.chunk_shape,
            "nr_servers": len(self.servers),
            "server_idx": server_idx,
        }
        return part_params


    def put(self, path_prefix, data, server_idx):
        name, info = self.servers[server_idx]
        path = (path_prefix + "{}" + self.base_dir).format('/' if self.base_dir[0] != '/' else "")
        url = 'http://%s:%d/%s' % (socket.inet_ntoa(info.address), info.port, path)
        url = url.encode()
        for repeat in range(self.repeats):
            session = self.sessions[server_idx]
            try:
                response = session.put(url, data=data, timeout=self.timeout)
                if not response.status_code == 200:
                    raise RuntimeError('Put failed with %d: %s' % (response.status_code, response.content))
                return
            # except requests.adapters.ConnectionError as err:
            #     self.logger.error("{}\n{}".format(str(err), str(err.__traceback__)))
            except response.ConnectTimeout:
                if repeat + 1 == self.repeats:
                    self.logger.error('Timeout attempting to put file: %s, after 3 retries, aborting' % url)
                    raise
                else:
                    self.logger.warn('Timeout attempting to put file: %s, retrying' % url)
            finally:
                try:
                    response.close()
                except:
                    pass

    def create_remote_part(self, server_idx, backend):
        data = json.dumps(self.part_dict(server_idx, backend)).encode()
        self.put('__part_pyramid_create', data, server_idx)
        print("Created remote part pyramid for server {}".format(server_idx))

    def get_tile_slices_from(self, x, y, tile_x, tile_y, frameSizeX, frameSizeY, frame, weights):
        xs = max(tile_x * self.tile_size - x, 0)
        xe = min((tile_x + 1) * self.tile_size - x, frameSizeX)
        xst = int(max(x - tile_x * self.tile_size, 0))
        xet = int(min(xst + (xe - xs), self.tile_size))

        ys = max((tile_y * self.tile_size) - y, 0)
        ye = min(((tile_y + 1) * self.tile_size) - y, frameSizeY)
        yst = int(max(y - tile_y * self.tile_size, 0))
        yet = int(min(yst + (ye - ys), self.tile_size))

        frame_slice = frame[xs:xe, ys:ye]
        weights_slice = weights[xs:xe, ys:ye]
        coords = [tile_x, tile_y, xst, xet, yst, yet]
        return frame_slice, weights_slice, coords

    def update_base_tiles_from_frame(self, x, y, frame, weights):
        frameSizeX, frameSizeY = frame.shape[:2]
        
        out_folder = os.path.join(self.base_dir, '0')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        if (x < 0) or (y < 0):
            raise ValueError('base tile origin positions must be >=0')
        
        tile_xs = range(int(np.floor(x / self.tile_size)), int(np.floor((x + frameSizeX) / self.tile_size) + 1))
        tile_ys = range(int(np.floor(y / self.tile_size)), int(np.floor((y + frameSizeY) / self.tile_size) + 1))

        self.n_tiles_x = max(self.n_tiles_x, max(tile_xs))
        self.n_tiles_y = max(self.n_tiles_y, max(tile_ys))

        for tile_x in tile_xs:
            for tile_y in tile_ys:
                server_idx = server_for_chunk(
                    tile_x, tile_y, 0, chunk_shape=self.chunk_shape, nr_servers=len(self.sessions)
                )
                frame_slice, weights_slice, coords = self.get_tile_slices_from(
                    x, y, tile_x, tile_y, frameSizeX, frameSizeY, frame, weights
                )
                self.send_tile_to_server(coords, frame_slice, weights_slice, server_idx)

        self.pyramid_valid = False

    def send_tile_to_server(self, coords, frame, weights, server_idx):
        data_dict = {
            "frame_shape": frame.shape,
            "frame_data": frame.ravel().tolist(),
            "weights_shape": weights.shape,
            "weights_data": weights.ravel().tolist(),
            "coords": coords,
        }
        data = json.dumps(data_dict).encode()
        self.put('__part_pyramid_update', data, server_idx)

    def finish_base_tiles(self):
        for server_idx in range(len(self.servers)):
            self.put('__part_pyramid_finish', "".encode(), server_idx)


def distributed_pyramid(out_folder, ds, xm, ym, mdh, split=False, skipMoveFrames=False, shiftfield=None,
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
        x positions of frames in ds. Raw stage positions in [um]. ImagePyramid
        origin will be at at minimum x, and offset to camera chip origin will
        be handled in SupertileDatasource tile_coords_um method.
        to the camera chip origin.
    ym : np.ndarray or PYME.Analysis.piecewiseMapping.piecewiseMap
        y positions of frames in ds. Raw stage positions in [um]. ImagePyramid
        origin will be at at minimum y, and offset to camera chip origin will
        be handled in SupertileDatasource tile_coords_um method.
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
    distributed : bool, optional
        whether to build a pyramid through distribution on the cluster.
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
    
    Notes
    -----
    Code is currently somewhat alpha in that the splitter functionality is 
    more or less untested, and we only get tile orientations right for primary
    cameras (i.e. when the stage is registered with multipliers to match the
    camera, rather than camera registered with orientation metadata to match it
    to the stage)

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

    #give some room at the edges
    bufSize = 0
    if correlate:
        bufSize = 300
    
    # to avoid building extra, empty tiles, the pyramid origin is the minimum
    # x and y position present in the tiles
    x0_pyramid, y0_pyramid = xps.min(), yps.min()
    xps -= x0_pyramid
    yps -= y0_pyramid

    # calculate origin independent of the camera ROI setting to store in
    # metadata for use in e.g. SupertileDatasource.DataSource.tile_coords_um
    x0_cam, y0_cam = get_camera_physical_roi_origin(mdh)
    x0 = x0_pyramid + mdh.voxelsize_nm.x / 1e3 * x0_cam
    y0 = y0_pyramid + mdh.voxelsize_nm.y / 1e3 * y0_cam

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
    
    # get splitter ROI coordinates in units of pixels
    ROIX1 = x0_cam + 1  # TODO - is splitter 1-indexed?
    ROIY1 = y0_cam + 1
    ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
    ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')
    
    if dark is None:
        dark = float(mdh.getOrDefault('Camera.ADOffset', 0))

    P = DistributedImagePyramid(
        out_folder, pyramid_tile_size, x0=x0, y0=y0, 
        pixel_size=mdh.getEntry('voxelsize.x'),
    )

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
            # TODO - account for orientation so this works for non-primary cams
            P.update_base_tiles_from_frame(x_i, y_i, d_weighted.squeeze(), weights.squeeze())

    P.finish_base_tiles()

    t2 = time.time()
    logger.debug('Added base tiles in %fs' % (t2 - t1))
    #P._occ.flush()
    logger.debug(time.time() - t2)
    logger.debug('Updating pyramid ...')
    P.update_pyramid() # TODO: make cluster-aware
    logger.debug(time.time() - t2)
    logger.debug('Done')

    with open(os.path.join(P.base_dir, 'metadata.json'), 'w') as f:
        f.write(P.mdh.to_JSON())
    
    return P    
