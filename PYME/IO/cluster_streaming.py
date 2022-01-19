from . import clusterIO
import socket
import queue
import threading
import logging

logger = logging.getLogger(__name__)
QUEUE_MAX_SIZE = 200 # ~10k frames

class Stream(object):
    """
    Class to handle spooling files in an asynchronous / non blocking manner to a single server
    
    TODO - use this in the spoolers as well??? It's a little cleaner than the existing code.
    """
    def __init__(self, server_address, server_port, dir_manager=None, filter=None):
        if not isinstance(server_address, str):
            server_address = socket.inet_ntoa(server_address)
            
        self._server_address = server_address
        self._server_port = server_port
        
        self._dir_manager = dir_manager
        self._filter = filter
        
        self._url = 'http://%s:%d/' % (server_address, server_port)

        self._put_queue = queue.Queue(QUEUE_MAX_SIZE)
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
                    
                if self._filter is not None:
                    # allow us to do, e.g. compression in the spooling thread
                    data = self._filter(data)

                dl = len(data)

                header = b'PUT /%s HTTP/1.1\r\nConnection: %s\r\nContent-Length: %d\r\n\r\n' % (
                    filename.encode(), connection, dl)
                
                logger.debug(header)
                
                self._socket.sendall(header)
                if dl > 0:
                    self._socket.sendall(data)
                    
                self._put_queue.task_done()
                self._rc_queue.put((filename, dl))
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
                    filename, dl = self._rc_queue.get(timeout=.1)
                    status, reason, msg = clusterIO._parse_response(fp)
                    if not status == 200:
                        logger.error(('Error spooling %s: Response %d - status %d' % (filename, status, reason)) + ' msg: ' + str(msg))
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
        
        
def distribution_function_round_robin(n_servers, i=None):  
    if i is None:
        # distribute at random
        import random
        return random.randrange(n_servers)

    return i % n_servers

class Streamer(object):
    """ Create a spooler instance which keeps one persistent connection to each node on the
    cluster and allows non-blocking streaming of data to these nodes

    """
    def __init__(self, serverfilter=clusterIO.local_serverfilter, servers=None, distribution_fcn=distribution_function_round_robin, filter=None):
        """
        Parameters
        ----------

        serverfilter : string
                The cluster identifier (when multiple clusters on one network)
        servers : list
                A manual list of servers. Usage is not recommended
        distribution_fcn : callable
                a function which assigns files to frames. Takes (at least) n_servers as an argument (provided by the spooler), but optionally any addition keyword
                arguments you pass to `put()`. In simple cases, this will be a frame number - e.g. i - see round_robin fcn above. distribution_fcn
                *must* provide defaults for all parameters other than n_servers so that sensible behaviour is achived when no value is provided.
        filter: callable
                a function which performs some operation on the data before it's saved. Typically format conversion and/or compression.
        """
        self._directory = clusterIO.get_dir_manager(serverfilter)

        if servers is None:
            # FIXME - avoid private _ns usage
            self.servers = [(socket.inet_ntoa(v.address), v.port) for k, v in self._directory._ns.get_advertised_services()]
        else:
            self.servers = servers

        assert len(self.servers) > 0, "No servers found for distribution. Make sure that cluster servers are running and can be reached from this device."
        
        self._n_servers = len(self.servers)
        self._streams =  [Stream(address, port, filter=filter) for address, port in self.servers]

        self._distribution_fcn = distribution_fcn

    #@classmethod
   
    
    def put(self, filename, data, **kwargs):
        """ Put, choosing a stream using the distribution function
        
        kwargs are used as arguments for the server distribution function. 

        """
        idx = self._distribution_fcn(n_servers = self._n_servers, **kwargs)

        self.put_stream(idx, filename, data)

    def put_stream(self, idx, filename, data):
        """ Put to a specific stream """
        self._streams[idx].put(filename, data)

    def close(self):
        for s in self._streams:
            s.close()


def n_cluster_servers(serverfilter):
    """ Convenience function for hard-coding distribution functions in, e.g. the clusterh5  case"""
    return len(clusterIO.get_dir_manager(serverfilter).dataservers)