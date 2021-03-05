import numpy as np
from PYME.IO.MetaDataHandler import(
    get_camera_roi_origin, get_camera_physical_roi_origin,
    load_json, NestedClassMDHandler,
)


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
from PYME.Analysis.tile_pyramid import (
    TILEIO_EXT, ImagePyramid, get_position_from_events, PZFTileIO
)


logger = logging.getLogger(__name__)


def server_for_chunk(x, y, z=0, chunk_shape=[8,8,1], nr_servers=1):
    """
    Returns the server responsible for the chunk of tiles at given (x, y, z).
    """
    server_id = np.floor(x / chunk_shape[0])
    server_id = server_id + np.floor(y / chunk_shape[1]) * nr_servers
    server_id = (server_id + np.floor(z / chunk_shape[2])) % nr_servers
    server_id = int(server_id)
    return server_id


def is_server_for_chunk(candidate_server, x, y, z=0, chunk_shape=[8,8,1], nr_servers=1):
    actual = server_for_chunk(x, y, chunk_shape=chunk_shape, nr_servers=nr_servers)
    return actual == candidate_server


def update_partial_pyramid(pyramid, sleep_time=1):
    """
    Payload execution of PartialPyramid's update thread.
    First updates base tiles until all frames are processed and then updates the
    higher pyramid layers.

    Parameters
    ----------

    pyramid : PartialPyramid
        the pyramid to be updated.
    sleep_time : float
        duration in which the update thread rests to reduce polling issues.
        Interpreted in units of seconds.
    """
    while not pyramid.all_tiles_received:
        # time.sleep(sleep_time)
        while not pyramid.update_queue.empty():
            data = pyramid.update_queue.get()
            pyramid.update_base_tile_from_request(data)
    pyramid.update_pyramid()


class PartialPyramid(ImagePyramid):
    """
    Subclass of ImagePyramid which supports distribution of pyramid files over a PYME
    cluster.

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
        self.nr_servers = nr_servers
        self.update_queue = queue.SimpleQueue()
        self.update_thread = threading.Thread(
            target=update_partial_pyramid, args=(self,), daemon=True
        )
        self.all_tiles_received = False

    @classmethod
    def build_from_request(self, filepath, part_dict):
        part_params = json.loads(part_dict.decode())
        return PartialPyramid(
            filepath,
            pyramid_tile_size=part_params["pyramid_tile_size"],
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

    def update_base_tile_from_request(self, data):
        """
        Updates a base tile with the input of a HTTP PUT request.

        Parameters:
        -----------
        data : bytes
            the bytes of a string from which a dictionary can be loaded (via JSON). Contains the input for update_base_tile_from_slices.
        """
        data_dict = json.loads(data.decode())
        frame_slice = np.asarray(data_dict["frame_data"])
        frame_slice = frame_slice.reshape(data_dict["frame_shape"])
        weights_slice = np.asarray(data_dict["weights_data"])
        weights_slice = weights_slice.reshape(data_dict["weights_shape"])
        coords = data_dict["coords"]
        self.update_base_tile_from_slices(0, coords, frame_slice, weights_slice)

    def get_tile_at_request(self, request):
        """
        Gets tile at the position specified by a dictionary request.

        Parameters:
        -----------
        request : dictionary
            Contains coordinates at keys ["layer", "x", "y"].
        
        Returns
        -------
        bytes
            tile numpy array converted to bytes.
        dtype
            dtype of the tile.
        tuple (of ints)
            shape of the tile. Supposed to be (tile_size, tile_size).
        
        Notes
        -----
        If the get_tile call at the same position would return None, this method returns
        the values of an empty array.
        This is to ensure there are bytes which can be sent via HTTP.
        """
        layer = int(request["layer"])
        x = int(request["x"])
        y = int(request["y"])
        tile = self.get_tile(layer, x, y)
        if tile is None:
            tile = np.asarray([], dtype=np.double)
        else:
            tile = tile.astype(np.double)
            assert tile.shape[0] == self.tile_size and tile.shape[1] == self.tile_size
        return tile.tobytes(), tile.dtype, tile.shape

    def get_coords_at_request(self, request):
        """
        Gets tile coords at the layer specified in the request dict.

        Parameters:
        -----------
        request : dictionary
            Contains coordinates at keys ["level"].
        
        Returns
        -------
        bytes
            of a JSON string of the coords.
        """
        level = int(request["level"])
        coords = self.get_layer_tile_coords(level)
        return json.dumps(coords).encode()

    def get_status_at_request(self):
        """
        Gets the current update status of this PartialPyramid. Used to determine whether this part
        has finished building.
        
        Returns
        -------
        status : dict
            indicates the update status via bool values at the keys
            ["base tiles done", "part pyramid done"].
        """
        status = {
            "base tiles done": not (self.all_tiles_received and self.update_queue.empty()),
            "part pyramid done": self.pyramid_valid,
        }
        return json.dumps(status).encode()

    def update_base_tile_from_slices(self, layer, coords, frame_slice, weights_slice):
        """
        Updates a base tile.
        
        Parameters
        ----------
        layer : int
            should be 0.
        coords : list or array
            specifies which tile and which part of the tile is updated.
        frame_slice : numpy array
            the part of the frame used to update the tile.
        weights_slice : numpy array
            the weights for this tile update.
        """

        [tile_x, tile_y, xst, xet, yst, yet] = coords
        acc_ = self._acc.get_tile(layer, tile_x, tile_y)
        occ_ = self._occ.get_tile(layer, tile_x, tile_y)

        if (acc_ is None) or (occ_ is None):
            acc_ = np.zeros([self.tile_size, self.tile_size], dtype=np.double)
            occ_ = np.zeros([self.tile_size, self.tile_size], dtype=np.double)

        acc_[xst:xet, yst:yet] += frame_slice
        occ_[xst:xet, yst:yet] += weights_slice
        self._acc.save_tile(layer, tile_x, tile_y, acc_)
        self._occ.save_tile(layer, tile_x, tile_y, occ_)

        self._clean_tiles(tile_x, tile_y)

    def rebuild_base(self):
        """
        Analoguous to _rebuild_base from the super class, but only rebuilds the tiles
        for which this server is responsible.
        """
        for xc, yc in self._occ.get_layer_tile_coords(0):
            if is_server_for_chunk(
                self.server_idx, xc, yc, z=0,
                chunk_shape=self.chunk_shape,
                nr_servers=self.nr_servers,
            ) and not self._imgs.tile_exists(0, xc, yc):
                occ = self._occ.get_tile(0, xc, yc) + 1e-9
                sf = 1.0 / occ
                sf[occ <= .1] = 0
                tile_ = self._acc.get_tile(0, xc, yc) * sf
                assert tile_.shape[0] == self.tile_size and tile_.shape[1] == self.tile_size
                self._imgs.save_tile(0, xc, yc, tile_)

    def make_new_layer(self, input_level):
        """
        Analoguous to _make_layer from the super class, but only makes the tiles
        for which this server is responsible.
        """
        from scipy import ndimage

        new_layer = input_level + 1
        tile_coords = self.get_layer_tile_coords(input_level)
        
        qsize = int(self.tile_size / 2)
        
        new_tile_coords = list(set([tuple(np.floor(np.array(tc) / 2).astype('i').tolist()) for tc in tile_coords]))
        layer_chunk_shape = np.asarray(self.chunk_shape) / pow(2, input_level)
        layer_chunk_shape[2] = 1

        for xc, yc in new_tile_coords:
            if is_server_for_chunk(
                self.server_idx, xc, yc, z=input_level,
                chunk_shape=layer_chunk_shape,
                nr_servers=self.nr_servers,
            ) and not self._imgs.tile_exists(new_layer, xc, yc):
                tile = np.zeros([self.tile_size, self.tile_size], dtype=np.double)
                
                NW = self.get_tile(input_level, 2 * xc, 2 * yc)
                if not NW is None:
                    tile[:qsize, :qsize] = ndimage.zoom(NW, .5)
                NE = self.get_tile(input_level, (2 * xc) + 1, (2 * yc))
                if not NE is None:
                    tile[qsize:, :qsize] = ndimage.zoom(NE, .5)
                SW = self.get_tile(input_level, (2 * xc), (2 * yc) + 1)
                if not SW is None:
                    tile[:qsize, qsize:] = ndimage.zoom(SW, .5)
                SE = self.get_tile(input_level, (2 * xc) + 1, (2 * yc) + 1)
                if not SE is None:
                    tile[qsize:, qsize:] = ndimage.zoom(SE, .5)
                assert tile.shape[0] == self.tile_size and tile.shape[1] == self.tile_size
                self._imgs.save_tile(new_layer, xc, yc, tile)

        is_not_final_part_layer = layer_chunk_shape.max() > 2
        return is_not_final_part_layer

    def update_pyramid(self):
        """
        Updates this PartialPyramid as far as its chunk_shape allows.
        """
        logger.debug("updating part pyramid")
        self.rebuild_base()
        inputLevel = 0
        
        while self.make_new_layer(inputLevel):
            logger.debug("Built level {}".format(inputLevel))
            inputLevel += 1
        logger.debug("Topmost level {}".format(inputLevel + 1))

        self.pyramid_valid = True
        self.depth = inputLevel
        self._imgs.flush()
        logger.debug("updating part pyramid done")



class DistributedImagePyramid(ImagePyramid):
    """
    Subclass of ImagePyramid which supports distribution of pyramid files over a PYME
    cluster.
    Implementation from the microscope side.
    """
    def __init__(
        self, storage_directory, pyramid_tile_size=256, mdh=None, 
        n_tiles_x=0, n_tiles_y=0, depth=0, x0=0, y0=0, 
        pixel_size=1, backend=PZFTileIO, servers=None, chunk_shape=[8,8,1], timeout=10, repeats=3
    ):
        super().__init__(
            storage_directory, pyramid_tile_size=pyramid_tile_size, mdh=mdh,
            n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, depth=depth, x0=x0, y0=y0, 
            pixel_size=pixel_size, backend=backend
        )
        self.chunk_shape = chunk_shape
        self.timeout = timeout
        self.repeats = repeats
        if servers is None:
            self.servers = [
                (
                    socket.inet_ntoa(v.address), v.port
                ) for k, v in clusterIO.get_ns().get_advertised_services()
            ]
        else:
            self.servers = [(address, servers[address]) for address in servers]
        assert len(self.servers) > 0, "No servers found for distribution. Make sure that cluster servers are running and can be reached from this device."
        self.sessions = [requests.Session() for _, _ in self.servers]
        for server_idx in range(len(self.servers)):
            self.create_remote_part(server_idx, backend)
        self._mdh["Pyramid.Servers"] = self.mdh_servers()
        self._mdh["Pyramid.ChunkShape"] = self.chunk_shape
        self._mdh["Pyramid.Timeout"] = self.timeout
        self._mdh["Pyramid.Repeats"] = self.repeats
        self._cached_level_coords = {}

    @classmethod
    def load_existing(cls, storage_directory):
        """ loads a DistributedImagePyramid from a given directory.

        Parameters
        ----------
        storage_directory : str
            root directory of a DistributedImagePyramid instance.

        Returns
        -------
        DistributedImagePyramid
            based on storage_directory contents.
        """

        mdh = load_json(os.path.join(storage_directory, 'metadata.json'))

        return DistributedImagePyramid(
            storage_directory,
            pyramid_tile_size=mdh['Pyramid.TileSize'],
            mdh=mdh,
            n_tiles_x=mdh["Pyramid.NTilesX"],
            n_tiles_y=mdh["Pyramid.NTilesY"],
            depth=mdh["Pyramid.Depth"],
            x0=mdh['Pyramid.x0'],
            y0=mdh['Pyramid.y0'],
            pixel_size=mdh["Pyramid.PixelSize"],
            servers=mdh["Pyramid.Servers"],
            chunk_shape=mdh["Pyramid.ChunkShape"],
            timeout=mdh["Pyramid.Timeout"],
            repeats=mdh["Pyramid.Repeats"],
        )

    def mdh_servers(self):
        """
        Converts the servers attribute into a dictionary for mdh.
        """
        servers_dict = {}
        for address, port in self.servers:
            servers_dict[address] = port
        return servers_dict

    def part_dict(self, server_idx, backend):
        """
        Returns a dict for initializing PartialPyramid on a cluster server.
        """
        part_params = {
            "pyramid_tile_size": self.tile_size,
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

    def make_url(self, path_prefix, server_idx):
        """
        Generates a URL to a specified cluster server for a request
        indicated by the path prefix.
        """
        address, port = self.servers[server_idx]
        path = (path_prefix + "{}" + self.base_dir).format(
            '/' if self.base_dir[0] != '/' else ""
        )
        url = 'http://%s:%d/%s' % (address, port, path)
        url = url.encode()
        return url

    def put(self, path_prefix, data, server_idx):
        """
        Sends a HTTP PUT request to a specified cluster server.
        """
        url = self.make_url(path_prefix, server_idx)
        for repeat in range(self.repeats):
            session = self.sessions[server_idx]
            try:
                response = session.put(url, data=data, timeout=self.timeout)
                if not response.status_code == 200:
                    raise RuntimeError('Put failed with %d: %s' % (response.status_code, response.content))
                return
            except response.ConnectTimeout:
                if repeat + 1 == self.repeats:
                    logger.error('Timeout attempting to put file: %s, after 3 retries, aborting' % url)
                    raise
                else:
                    logger.warn('Timeout attempting to put file: %s, retrying' % url)
            finally:
                try:
                    response.close()
                except:
                    pass

    def get(self, path_prefix, request_data, server_idx):
        """
        Sends a HTTP GET request to a specified cluster server.
        """
        url = self.make_url(path_prefix, server_idx)
        for repeat in range(self.repeats):
            session = self.sessions[server_idx]
            try:
                response = session.get(url, params=request_data, timeout=self.timeout)
                if not response.status_code == 200:
                    raise RuntimeError('Put failed with %d: %s' % (response.status_code, response.content))
                return response
            except response.ConnectTimeout:
                if repeat + 1 == self.repeats:
                    logger.error('Timeout attempting to put file: %s, after 3 retries, aborting' % url)
                    raise
                else:
                    logger.warn('Timeout attempting to put file: %s, retrying' % url)
            finally:
                try:
                    response.close()
                except:
                    pass
        return None

    def get_tile(self, layer, x, y):
        if layer >= self.top_part_layer() or self._imgs.tile_exists(layer, x, y):
            return self._imgs.get_tile(layer, x, y)
        elif layer < self.top_part_layer():
            tile = self.get_tile_from_server(layer, x, y)
            if tile is not None:
                self._imgs.save_tile(layer, x, y, tile)
            return tile

    def get_layer_tile_coords(self, level):
        if level >= self.top_part_layer():
            return self._imgs.get_layer_tile_coords(level)
        elif str(int(level)) in self._cached_level_coords:
            return self._cached_level_coords[str(int(level))]
        layer_coords = self.get_layer_tile_coords_from_servers(level)
        self._cached_level_coords[str(int(level))] = layer_coords
        return layer_coords

    def create_remote_part(self, server_idx, backend):
        """
        Initializes PartialPyramid a on the specified server.
        """
        data = json.dumps(self.part_dict(server_idx, backend)).encode()
        self.put('__part_pyramid_create', data, server_idx)
        logger.debug("Created remote part pyramid for server {}".format(server_idx))

    def get_tile_slices_from(
        self, x, y, tile_x, tile_y, frameSizeX, frameSizeY, frame, weights
    ):
        """
        Extracts the update coordinates and slices from an update frame and its weights.
        Essentially prepares the input for PartialPyramid.update_base_tile_from_slices().
        """
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
        
        tile_xs = range(
            int(
                np.floor(x / self.tile_size)), int(np.floor((x + frameSizeX) / self.tile_size) + 1
            )
        )
        tile_ys = range(
            int(
                np.floor(y / self.tile_size)), int(np.floor((y + frameSizeY) / self.tile_size) + 1
            )
        )

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
                self.update_tile_on_server(coords, frame_slice, weights_slice, server_idx)

        self.pyramid_valid = False

    def update_tile_on_server(self, coords, frame, weights, server_idx):
        """
        Sends the output from get_tile_slices_from() to the responsible cluster
        server.
        """
        data_dict = {
            "frame_shape": frame.shape,
            "frame_data": frame.ravel().tolist(),
            "weights_shape": weights.shape,
            "weights_data": weights.ravel().tolist(),
            "coords": coords,
        }
        data = json.dumps(data_dict).encode()
        self.put('__part_pyramid_update_tile', data, server_idx)

    def finish_base_tiles(self):
        """
        Notifies all cluster servers that all frames have been processed on the
        microscope end.
        """
        for server_idx in range(len(self.servers)):
            self.put('__part_pyramid_finish', "".encode(), server_idx)

    def get_status_from_server(self, server_idx):
        """
        Requests the update status of the PartialPyramid on a given cluster server.
        """
        response = self.get("__part_pyramid_status", {}, server_idx)
        return json.loads(response.content.decode())

    def get_tile_from_server(self, layer, x, y):
        """
        Loads a tile which is located on a cluster server.
        """
        layer_chunk_shape = np.asarray(self.chunk_shape) / pow(2, layer)
        layer_chunk_shape[2] = 1
        server_idx = server_for_chunk(x, y, chunk_shape=layer_chunk_shape, nr_servers=len(self.servers))
        response = self.get("__part_pyramid_tile", {"layer": layer, "x": x, "y": y}, server_idx)
        tile_data = response.content
        tile = np.frombuffer(tile_data, dtype=np.double)
        if len(tile) == 0:
            return None
        tile = tile.reshape((self.tile_size, self.tile_size))
        return tile

    def get_layer_tile_coords_from_servers(self, level):
        """
        Loads the tile coords at a specified level from the cluster.
        """
        aggregate = []
        for server_idx in range(len(self.servers)):
            response = self.get('__part_pyramid_coords', {"level": level}, server_idx)
            tile_coords = json.loads(response.content.decode())
            aggregate.append(tile_coords)
        return tile_coords

    def rebuild_base(self):
        """
        Synchronization step between the microscope and the cluster.
        This method returns when all PartialPyramids for this DistributedImagePyramid have been built.
        """
        unfinished_servers = set([i for i in range(len(self.servers))])
        while len(unfinished_servers) > 0:
            finished = []
            # time.sleep(len(unfinished_servers))
            time.sleep(1)
            for server_idx in unfinished_servers:
                status = self.get_status_from_server(server_idx)
                if status["part pyramid done"]:
                    finished.append(server_idx)
            for server_idx in finished:
                unfinished_servers.remove(server_idx)
        logger.debug("All remote parts built")

    def top_part_layer(self):
        """
        Returns the top layer a PartialPyramid has built given chunk_shape.
        """
        layer_chunk_shape = np.asarray(self.chunk_shape)
        level = 0
        while layer_chunk_shape.max() > 1:
            level = level + 1
            layer_chunk_shape = layer_chunk_shape / 2
        return level

    def update_pyramid(self):
        """
        Builds the final layers once the lower levels have been built on the cluster.
        The topmost layers on the cluster are aggregated on the microscope and
        then used to make the final layers.
        """
        self.rebuild_base()
        inputLevel = self.top_part_layer()
        tile_coords = self.get_layer_tile_coords_from_servers(inputLevel)
        for x, y in tile_coords:
            tile = self.get_tile_from_server(inputLevel, x, y)
            if tile is not None:
                self._imgs.save_tile(inputLevel, x, y, tile)
        logger.debug("Aggregated parts of level {}".format(inputLevel))

        while self._make_layer(inputLevel) > 1:
            inputLevel += 1
            logger.debug("Built level {}".format(inputLevel))
        logger.debug("Topmost level: {}".format(inputLevel + 1))

        self.pyramid_valid = True
        self.depth = inputLevel
        self._imgs.flush()


def distributed_pyramid(
    out_folder, ds, xm, ym, mdh, split=False, skipMoveFrames=False,
    shiftfield=None, mixmatrix=[[1., 0.], [0., 1.]],
    correlate=False, dark=None, flat=None, pyramid_tile_size=256
):
    """Create a distributed pyramid through PYMECluster.

    Parameters
    ----------
    out_folder : str
        directory to save pyramid tiles(/directories). The same folder will be
        created on the cluster servers.
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
    DistributedImagePyramid
        coalesced/averaged/etc multilevel DistributedImagePyramid instance
    
    Notes
    -----
    Code is currently somewhat alpha in that the splitter functionality is 
    more or less untested, and we only get tile orientations right for primary
    cameras (i.e. when the stage is registered with multipliers to match the
    camera, rather than camera registered with orientation metadata to match it
    to the stage).

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

    logger.debug('Updating base tiles ...')
    
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
    logger.debug('Updated base tiles in %fs' % (t2 - t1))
    #P._occ.flush()
    logger.debug(time.time() - t2)
    logger.debug('Updating pyramid ...')
    P.update_pyramid() # TODO: make cluster-aware
    logger.debug(time.time() - t2)
    logger.debug('Done')

    with open(os.path.join(P.base_dir, 'metadata.json'), 'w') as f:
        f.write(P.mdh.to_JSON())
    
    return P    


def create_distributed_pyramid_from_dataset(filename, outdir, tile_size=128, **kwargs):
    from PYME.IO import image
    dataset = image.ImageStack(filename=filename)

    xm, ym = get_position_from_events(dataset.events, dataset.mdh)

    p = distributed_pyramid(outdir, dataset.data, xm, ym, dataset.mdh, pyramid_tile_size=tile_size)

    return p


if __name__ == '__main__':
    import sys
    import time
    input_stack, output_dir = sys.argv[1:]
    t1 = time.time()
    create_distributed_pyramid_from_dataset(input_stack, output_dir)
    logger.debug(time.time() - t1)
 