import numpy as np
from PYME.IO.MetaDataHandler import get_camera_roi_origin, get_camera_physical_roi_origin, load_json, NestedClassMDHandler

import threading
import queue
import os
import time
import six
import socket
import requests
import logging
import json

from PYME.IO import clusterIO, PZFFormat
from PYME.Analysis.tile_pyramid import TILEIO_EXT, ImagePyramid, get_position_from_events, PZFTileIO, ClusterPZFTileIO

logger = logging.getLogger(__name__)
MAXLINE = 65536


def server_for_chunk(x, y, z=0, chunk_shape=[8,8,1], nr_servers=1):
    """
    Returns the server responsible for the chunk of tiles at given (x, y, z).
    """
    server_id = np.floor(x / chunk_shape[0])
    server_id = server_id + np.floor(y / chunk_shape[1]) * nr_servers
    server_id = (server_id + np.floor(z / chunk_shape[2])) % nr_servers
    server_id = int(server_id)
    return server_id

class PartialPyramid(ImagePyramid):
    """
    Subclass of ImagePyramid which supports distribution of pyramid files over a PYME
    cluster.

    Implementation on the server side
    
    Note: we don't want to write metadata for the partial pyramids, meaning that they don't
    need all of the parameters
    
    """
    def __init__(self, storage_directory, pyramid_tile_size=256, backend=PZFTileIO, chunk_shape=[8,8,1]):
        ImagePyramid.__init__(self,storage_directory, pyramid_tile_size=pyramid_tile_size, backend=backend)
        
        logger.debug('Creating new pyramid in directory %s' % storage_directory)
        
        self.chunk_shape = chunk_shape
        self.update_queue = queue.Queue()
        
        self.all_tiles_received = False

        self._update_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._update_thread.start()

    @classmethod
    def from_request(cls, filepath, body):
        params = json.loads(body.decode())
        
        return cls(filepath, **params)
                    
    def queue_base_tile_update(self, data, query):
        import json
        data, _ = PZFFormat.loads(data)
        data = data.squeeze()
        weights = query.get('weights', ['auto',])[0]
        
        tile_x, tile_y = int(query['x'][0]), int(query['y'][0])
        tile_offset = (int(query.get('tox', [0,])[0]), int(query.get('toy', [0,])[0]))
        frame_offset = (int(query.get('fox', [0,])[0]), int(query.get('foy', [0,])[0]))
        
        frame_shape = (int(query['fsx'][0]),int(query['fsy'][0]))
        
        tile_data = tile_x, tile_y, data, weights, tile_offset, frame_offset, frame_shape
        
        #logger.debug('putting data on update_queue, shape=%s' % str(data.shape))
        self.update_queue.put(tile_data)
        
    def _poll_loop(self):
        logger.debug('starting polling loop')
        while not self.all_tiles_received:
            # time.sleep(sleep_time)
            try:
                # we let queue.get() block until there is data in the queue, albeit with a timeout
                # so that we can
                tile_data = self.update_queue.get(timeout=.1)
                
                #logger.debug('got data from update queue')

                self.update_base_tile(*tile_data)
                self.pyramid_valid = False
            except queue.Empty:
                # build the pyramid on the fly if we're not busy processing updates
                if not self.pyramid_valid:
                    logger.debug('Rebuilding base')
                    #self._rebuild_base()
                    self.update_pyramid()
                pass
            except:
                logger.exception('Unexpected error in polling loop:')
                raise
        
        self.update_pyramid()
        

    def finalise(self):
        self.all_tiles_received = True
        
        # this will block until the part-pyramid is complete, making the calling function block as well.
        self._update_thread.join()
    
    def update_pyramid(self):
        """
        Updates this PartialPyramid as far as its chunk_shape allows.
        
        TODO - refactor base update_pyramid to allow max_depth as a parameter.
        """
        logger.debug("updating part pyramid")
        self._rebuild_base()
        inputLevel = 0
        
        max_depth = np.log2(self.chunk_shape[:2]).min()
        
        while (inputLevel < max_depth) and (self._make_layer(inputLevel) > 1):
            logger.debug("Built level {}".format(inputLevel))
            inputLevel += 1
        
        logger.debug("Topmost level {}".format(inputLevel + 1))

        self.pyramid_valid = True
        self.depth = inputLevel
        self._imgs.flush()
        logger.debug("updating part pyramid done")


class TileSpooler(object):
    """
    Class to handle spooling tiles to a single server
    
    TODO - use this in the spoolers as well??? It's a little cleaner than the existing code.
    """
    def __init__(self, server_address, server_port, dir_manager=None):
        if not isinstance(server_address, str):
            server_address = socket.inet_ntoa(server_address)
            
        self._server_address = server_address
        self._server_port = server_port
        
        self._dir_manager = dir_manager
        
        self._url = 'http://%s:%d/' % (server_address, server_port)

        self._put_queue = queue.Queue()
        self._rc_queue = queue.Queue()
        #self._last_flush_time = time.time()
        self._socket = None
        self._alive = True
        
        self._connect()
        
        self._t_send = threading.Thread(target=self._send_loop)
        self._t_send.daemon = True
        self._t_send.start()

        self._t_recv = threading.Thread(target=self._recv_loop)
        self._t_recv.daemon = True
        self._t_recv.start()
        
    def _connect(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.settimeout(30)
        
        self._socket.connect((self._server_address, self._server_port))
    
    def finalize(self, filename=None):
        """
        Stop our threads and tidy up

        """
        if filename:
            #if we provide a sentinal file, put that
            self.put(filename, None)
            self._t_send.join(2)
        
        
        #self._alive = False
        
    def close(self):
        #TODO - wait on the put queue to empty?
        self._alive = False
        self._t_send.join()
        self._t_recv.join()
        
        self._socket.close()
        
    def put(self, filename, data):
        """Add a file to the queue to be put
        
        calling put with data == None is a sentinel that no more data will follow, and that keep-alive
        should be switched off. This should normaly be accompanied by a filename = '__part_pyramid_finish/<pyramid_dir>'
        
        """
        self._put_queue.put((filename, data))
    
    def _send_loop(self):
        """
        Loop which runs in it's own thread, pushing chunks as they come in. Because we now run sending and recieving
        in separate threads, and keep the socket open for the entire duration, we don't need to batch puts.

        """
        connection = b'keep-alive'
        
        while self._alive:
            try:
                filename, data = self._put_queue.get(timeout=.1)

                if data is None:
                    connection = b'close'
                    data = b''
                    
                dl = len(data)

                header = b'PUT /%s HTTP/1.1\r\nConnection: %s\r\nContent-Length: %d\r\n\r\n' % (
                    filename.encode(), connection, dl)
                
                logger.debug(header)
                
                self._socket.sendall(header)
                if dl > 0:
                    self._socket.sendall(data)
                    
                self._put_queue.task_done()

                #datalen += dl
                #nChunksSpooled += 1
                #nChunksRemaining -= 1
                
            except queue.Empty:
                pass
                
            
    def _recv_loop(self):
        fp = self._socket.makefile('rb', 65536)
        
        try:
            while self._alive:
                try:
                    filename = self._rc_queue.get(timeout=.1)
                    status, reason, msg = clusterIO._parse_response(fp)
                    if not status == 200:
                        logger.error(('Error spooling %s: Response %d - status %d' % (filename, i, status)) + ' msg: ' + str(msg))
                    else:
                        if self._dir_manager:
                            # register file in cluster directory
                            fn = filename.split('?')[0]
                            url = self.url_ + fn
                            self._dir_manager.register_file(fn, url, dl)
                            
                except queue.Empty:
                    pass
                
                #TODO - more detailed error handling - e.g. socket timeouts
                
        finally:
            fp.close()
        
        
        


class DistributedImagePyramid(ImagePyramid):
    """
    Subclass of ImagePyramid which supports distribution of pyramid files over a PYME
    cluster.
    Implementation from the microscope side.
    """
    def __init__(self, storage_directory, pyramid_tile_size=256, mdh=None, n_tiles_x=0, n_tiles_y=0, depth=0, x0=0, y0=0,
                 pixel_size=1, backend=ClusterPZFTileIO, servers=None, chunk_shape=[8,8,1], timeout=10, repeats=3):
        
        ImagePyramid.__init__(self,storage_directory, pyramid_tile_size=pyramid_tile_size, mdh=mdh,
                              n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, depth=depth, x0=x0, y0=y0,
                              pixel_size=pixel_size, backend=backend)
        
        self.chunk_shape = chunk_shape
        self.timeout = timeout
        self.repeats = repeats
        
        if servers is None:
            self.servers = [(socket.inet_ntoa(v.address), v.port) for k, v in clusterIO.get_ns().get_advertised_services()]
        else:
            self.servers = [(address, servers[address]) for address in servers]
            
        assert len(self.servers) > 0, "No servers found for distribution. Make sure that cluster servers are running and can be reached from this device."
        
        self._tile_spoolers = [TileSpooler(address, port) for address, port in self.servers]
        
        for server_idx in range(len(self.servers)):
            self.create_remote_part(server_idx, backend)
        
        # TODO - do we need to put the chunk shape in the metadata?
        #self._mdh["Pyramid.ChunkShape"] = self.chunk_shape
        #self._cached_level_coords = {}


    def create_remote_part(self, server_idx, backend):
        """
        Initializes PartialPyramid a on the specified server.
        """
        data = json.dumps({"pyramid_tile_size": self.tile_size,"chunk_shape": self.chunk_shape,}).encode()
        
        address, port = self.servers[server_idx]
        p_dir = self.base_dir.lstrip('/')
        url = f'http://{address}:{port}/__pyramid_create/{p_dir}'

        # TODO - do this through the tile spoolers to hide latency? (currently serialised over servers)
        r = requests.put(url, data=data)
        if not r.status_code == 200:
            msg = 'Put failed with %d: %s' % (r.status_code, r.content)
            r.close()
            raise RuntimeError(msg)
        r.close()
        
        logger.debug("Created remote part pyramid for server {}".format(server_idx))


    def _ensure_layer_directory(self, layer_num=0):
        # directories get created automatically on the cluster, we don't need to do anything here.
        pass
    
    
    def update_base_tile(self, tile_x, tile_y, data, weights, tile_offset=(0,0), frame_offset=(0,0), frame_shape=None):
        """
        Over-ridden version of update_base_tile which causes this to be called on the server rather than the client
        
        In practice it simply adds each chunk to a queue of chunks that get pushed asynchronously in multiple threads
        (one for each server).
        """
        import json
        server_idx = server_for_chunk(tile_x, tile_y, chunk_shape=self.chunk_shape, nr_servers=len(self.servers))
        
        if weights is not 'auto':
            raise RuntimeError('Distributed pyramid only supports auto weights')
        
        fn = f'__pyramid_update_tile/{self.base_dir}?x={tile_x}&y={tile_y}&' + \
             f'tox={tile_offset[0]}&toy={tile_offset[1]}&fox={frame_offset[0]}&foy={frame_offset[1]}&' + \
             f'fsx={frame_shape[0]}&fsy={frame_shape[1]}'
        
        self._tile_spoolers[server_idx].put(fn,PZFFormat.dumps(data))
             
    
    def finish_base_tiles(self):
        """
        Notifies all cluster servers that all frames have been processed on the
        microscope end. Blocks until the servers have finished constructing their partial pyramids
        """
        
        for ts in self._tile_spoolers:
            ts.finalize('__pyramid_finish/' + self.base_dir.lstrip('/'))
            
        for ts in self._tile_spoolers:
            # put this in a separate loop to the above as .close() joins the pushing threads, which
            # would serialise computation (rather than having it triggered in all threads)
            ts.close()

    
    @property
    def partial_pyramid_depth(self):
        return int(np.log2(self.chunk_shape[:2]).min())

    
    def update_pyramid(self):
        """
        Builds the final layers once the lower levels have been built on the cluster.
        The topmost layers on the cluster are aggregated on the microscope and
        then used to make the final layers.
        """
        
        if not self.pyramid_valid:
            #only rebuild if we haven't already
            inputLevel = self.partial_pyramid_depth
    
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
    
    TODO - this largely duplicates the corresponding function in tile_pyramid => refactor

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
    
    # get splitter ROI coordinates in units of pixels
    ROIX1 = x0_cam + 1  # TODO - is splitter 1-indexed?
    ROIY1 = y0_cam + 1
    ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
    ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')
    
    if dark is None:
        dark = float(mdh.getOrDefault('Camera.ADOffset', 0))

    P = DistributedImagePyramid(out_folder, pyramid_tile_size, x0=x0, y0=y0, pixel_size=mdh.getEntry('voxelsize.x'),)

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

            # TODO - account for orientation so this works for non-primary cams
            P.update_base_tiles_from_frame(x_i, y_i, d)

    P.finish_base_tiles()

    t2 = time.time()
    logger.debug('Updated base tiles in %fs' % (t2 - t1))
    #P._occ.flush()
    logger.debug(time.time() - t2)
    logger.debug('Updating pyramid ...')
    P.update_pyramid() # TODO: make cluster-aware
    logger.debug(time.time() - t2)
    logger.debug('Done')

    
    clusterIO.put_file('/'.join([P.base_dir, 'metadata.json']), P.mdh.to_JSON())
    
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
    # import profile, cProfile, pstats
    input_stack, output_dir = sys.argv[1:]
    t1 = time.time()
    # profiler = cProfile.Profile()
    # profiler.enable()
    create_distributed_pyramid_from_dataset(input_stack, output_dir)
    # profiler.disable()
    logger.debug(time.time() - t1)
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats("distributed_pyramid")
    # stats.dump_stats(filename="profile.dmp")