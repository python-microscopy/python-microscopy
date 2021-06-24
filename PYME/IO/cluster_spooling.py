from . import clusterIO
import socket
import queue
import threading
import logging

logger = logging.getLogger(__name__)

class Stream(object):
    """
    Class to handle spooling files in an asynchronous / non blocking manner to a single server
    
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
        
        
def distribution_function_round_robin(i, n_servers):
    return i % n_servers

class Spooler(object):
    """ Create a spooler instance which keeps one persistent connection to each node on the
    cluster and allows non-blocking streaming of data to these nodes

    """
    def __init__(self, servers=None, distribution_fcn=distribution_function_round_robin):
        if servers is None:
            self.servers = [(socket.inet_ntoa(v.address), v.port) for k, v in clusterIO.get_ns().get_advertised_services()]
        else:
            self.servers = servers

        assert len(self.servers) > 0, "No servers found for distribution. Make sure that cluster servers are running and can be reached from this device."
        
        self._n_servers = len(self.servers)
        self._streams =  [Stream(address, port) for address, port in self.servers]

        self._distribution_fcn = distribution_fcn

    def put(self, filename, data, **kwargs):
        """ Put, choosing a stream using the distribution function"""
        idx = self._distribution_fcn(n_servers = self._n_servers, **kwargs)

        self.put_stream(idx, filename, data)

    def put_stream(self, idx, filename, data):
        """ Put to a specific stream """
        self._streams[idx].put(filename, data)

    def close(self):
        for s in self._streams:
            s.close()