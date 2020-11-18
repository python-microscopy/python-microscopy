# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:12:18 2016

@author: david

This is a very simple HTTP server which allows data to be saved to the server using PUT

Security note:
==============

The code as it stands lets any client write arbitrary data to the server. 
The only concessions we make to security are:

a) paths are vetted to make sure that they reside under out root (i.e. no escape through .. etc)
b) we enforce a write-once scheme (if a file exists it can't be overridden)

This protects against people accidentally or maliciously altering data, or discovering 
server settings, but leaves us open to denial of service type attacks in which
a malicious client could fill up our storage.

THE CODE AS IT STANDS SHOULD ONLY BE USED ON A TRUSTED NETWORK

TODO: Add some form of authentication. Needs to be low overhead (e.g. digest based)
"""
from PYME import config
from PYME.misc.computerName import GetComputerName
compName = GetComputerName()
import os
import posixpath
import urllib
import mimetypes

#make sure we set up our logging before anyone elses does
import logging    
dataserver_root = config.get('dataserver-root')
if dataserver_root:
    log_file = '%s/LOGS/%s/PYMEDataServer.log' % (dataserver_root, compName)
        
    logging.basicConfig(filename =log_file, level=logging.DEBUG)
    logger = logging.getLogger('')
else:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('')
    
#now do all the normal imports
    
from wsgiref.simple_server import make_server

#from StringIO import StringIO
#import shutil
#import urllib
import sys
import json
import PYME.misc.pyme_zeroconf as pzc
try:
    # noinspection PyCompatibility
    from urlparse import urlparse
except ImportError:
    #py3
    # noinspection PyCompatibility
    from urllib.parse import urlparse
    
import requests
import socket
#import fcntl
import threading
import datetime
import time
from PYME.IO.FileUtils.nameUtils import get_service_name

LOG_REQUESTS = False#True

startTime = datetime.datetime.now()
global_status = {}

textfile_locks = {}
def getTextFileLock(filename):
    try:
        return textfile_locks[filename]
    except KeyError:
        textfile_locks[filename] = threading.Lock()
        return textfile_locks[filename]


from collections import OrderedDict

class _LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


_dirCache = _LimitedSizeDict(size_limit=100)
_dirCacheTimeout = 1

def ResponseOK(content='', status='200 OK', headers=[]):
    return status, headers, content

def ResponseNotFound(content='', status='404 Not Found', headers=[]):
    headers += [('Content-type', 'text/plain')]
    return status, headers, content
    
def ResponseNotAllowed(content='', status='405 Method Not Allowed', headers=[]):
    headers += [('Content-type', 'text/plain')]
    return status, headers, content

server_config = {
    'bandwidthTesting':False,
    'logrequests':False
}

class dataserver(object):
    bandwidthTesting = False
    logrequests = False
    
    def __call__(self, environ, start_response):
        self.bandwidthTesting = server_config['bandwidthTesting']
        
        path = environ['PATH_INFO']
        try:
            data = environ['wsgi.input'].read(environ['CONTENT_LENGTH'])
        except KeyError:
            data = None

        status, headers, self.body = getattr(self, 'do_' + environ['REQUEST_METHOD'])(path, data, environ)
        if isinstance(self.body, str):
            headers += [('Content-Length', str(len(self.body))), ]

        start_response(status, headers)

        return [self.body]


    def _aggregate_txt(self, path, data):
        """
        support for results aggregation (append into a file). This 'method' does a dumb append to the file on disk,
        meaning that it will require reasonably carefully formatted input (with, e.g., a trailing newlines on each
        chunk if results are in .csv
        This is mostly intended for output in csv or similar formats, where chunks can be 'cat'ed together.

        NOTE: there is no guarantee of ordering, so the record format should contain enough information to re-order
        the data if necessary
        """
        path = self.translate_path(path.lstrip('/')[len('__aggregate_txt'):])

        #append the contents of the put request
        with getTextFileLock(path):
            #lock so that we don't corrupt the data by writing from two different threads
            #TODO ?? - keep a cache of open files
            with open(path, 'ab') as f:
                f.write(data)

        return ResponseOK()
        

    def _aggregate_h5r(self, path, data):
        """
        Support for results aggregation into an HDF5 file, using pytables.
        We treat any path components after the .h5r as locations within the file (ie table names).
        e.g. /path/to/data.h5r/<tablename>
        A few special cases / Table names are accommodated:

        MetaData: assumes we have sent PYME metadata in json format and saves to the file using the appropriate metadatahandler
        No table name: assumes we have a fitResults object (as returned by remFitBuf and saves to the the appropriate tables (as HDF task queue would)
        """
        import numpy as np
        from io import BytesIO
        from six.moves import cPickle
        from PYME.IO import MetaDataHandler
        from PYME.IO import h5rFile

        path = self.translate_path(path.lstrip('/')[len('__aggregate_h5r'):])
        filename, tablename = path.split('.h5r')
        filename += '.h5r'

        #logging.debug('opening h5r file')
        with h5rFile.openH5R(filename, 'a') as h5f:
            if tablename == '/MetaData':
                mdh_in = MetaDataHandler.CachingMDHandler(json.loads(data))
                h5f.updateMetadata(mdh_in)
            elif tablename == '':
                #legacy fitResults structure
                fitResults = cPickle.loads(data)
                h5f.fileFitResult(fitResults)
            else:
                try:
                    #try to read data as if it was numpy binary formatted
                    data = np.load(BytesIO(data))
                except IOError:
                    #it's not numpy formatted - try json
                    import pandas as pd
                    #FIXME!! - this will work, but will likely be really slow!
                    data = pd.read_json(data).to_records(False)

                #logging.debug('adding data to table')
                h5f.appendToTable(tablename.lstrip('/'), data)
                #logging.debug('added data to table')

        #logging.debug('left h5r file')
        return ResponseOK()

    def _doAggregate(self, path, data):
        # TODO - add authentication/checks for aggregation. Files which still allow appends should not be duplicated.
        if path.lstrip('/').startswith('__aggregate_txt'):
            return self._aggregate_txt(path, data)
        elif path.lstrip('/').startswith('__aggregate_h5r'):
            return self._aggregate_h5r(path, data)


    def do_PUT(self, path, data, environ):
        if self.bandwidthTesting:
            #just read file and dump contents
            return ResponseOK()

        if path.lstrip('/').startswith('__aggregate'):
            #paths starting with __aggregate are special, and trigger appends to an existing file rather than creation
            #of a new file.
            return self._doAggregate(path, data)


        path = self.translate_path(path)

        if os.path.exists(path):
            #Do not overwrite - we use write-once semantics
            return ResponseNotAllowed("File already exists")
        else:
            dir, file = os.path.split(path)
            if not os.path.exists(dir):
                os.makedirs(dir)

            query = urlparse.parse_qs(environ['QUERY_STRING'])

            if file == '':
                #we're  just making a directory
                pass
            elif 'MirrorSource' in query.keys():
                #File content is not in message content. This computer should
                #fetch the results from another computer in the cluster instead
                #used for online duplication

                r = requests.get(query['MirrorSource'][0], timeout=1)

                with open(path, 'wb') as f:
                    f.write(r.content)

            else:
                #the standard case - use the contents of the put request
                with open(path, 'wb') as f:
                    f.write(data)

            return ResponseOK()

    def do_HEAD(self, path, data, environ):
        """Serve a HEAD request."""
        return self.do_GET(path, data, environ, head=True)
        

    def get_status(self):
        from PYME.IO.FileUtils.freeSpace import disk_usage

        status = {}
        status.update(global_status)

        total, used, free = disk_usage(os.getcwd())
        status['Disk'] = {'total':total, 'used':used, 'free':free}
        status['Uptime'] = str(datetime.datetime.now() - startTime)

        try:
            import psutil

            status['CPUUsage'] = psutil.cpu_percent(interval=.1, percpu=True)
            status['MemUsage'] = psutil.virtual_memory()._asdict()
        except ImportError:
            pass

        content = json.dumps(status)
        encoding = sys.getfilesystemencoding()
        return ResponseOK(content, headers=[("Content-type", "application/json; charset=%s" % encoding),])
        
    
    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.

        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)

        """
        # abandon query parameters
        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        # Don't forget explicit trailing slash when normalizing. Issue17324
        trailing_slash = path.rstrip().endswith('/')
        path = posixpath.normpath(urllib.unquote(path))
        words = path.split('/')
        words = filter(None, words)
        path = os.getcwd()
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir): continue
            path = os.path.join(path, word)
        if trailing_slash:
            path += '/'
        return path

    def guess_type(self, path):
        """Guess the type of a file.

        Argument is a PATH (a filename).

        Return value is a string of the form type/subtype,
        usable for a MIME Content-type header.

        The default implementation looks the file's extension
        up in the table self.extensions_map, using application/octet-stream
        as a default; however it would be permissible (if
        slow) to look inside the data to make a better guess.

        """

        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']

    if not mimetypes.inited:
        mimetypes.init() # try to read system mime.types

    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream', # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
    })

    def do_GET(self, path, data, environ, head=False):
        """Common code for GET and HEAD commands.

        This sends the response code and MIME headers.

        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.

        """
        if path.lstrip('/') == '__status':
            return self.get_status()


        tpath = self.translate_path(path)
        f = None

        if os.path.isdir(tpath):
            return self.list_directory(tpath)
            
        ctype = self.guess_type(tpath)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(tpath, 'rb')
        except IOError:
            return ResponseNotFound("File not found - %s, [%s]" % (path, tpath))
        try:
            #fs = os.fstat(f.fileno())
            headers = [("Content-type", ctype),]
                       #("Last-Modified", self.date_time_string(fs.st_mtime))]
            #self.send_header("Content-Length", str(fs[6]))
                       
            if head:
                return ResponseOK(headers=headers)
            else:
                return ResponseOK(f.read(), headers=headers)
                
        finally:
            f.close()
            

    def list_directory(self, path):
        """Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().

        """
        curTime = time.time()
        
        if not path.endswith('/'):
            path += '/'

        try:
            js_dir, expiry = _dirCache[path]
            if expiry < curTime: raise RuntimeError('Expired')
        except (KeyError, RuntimeError):
            try:
                list = os.listdir(path)
            except os.error:
                return ResponseNotAllowed("No permission to list directory")

            list.sort(key=lambda a: a.lower())

            #displaypath = cgi.escape(urllib.unquote(self.path))
            l2 = []
            for l in list:
                if os.path.isdir(os.path.join(path, l)):
                    l2.append(l + '/')
                else:
                    l2.append(l)

            js_dir = json.dumps(l2)
            _dirCache[path] = (js_dir, curTime + _dirCacheTimeout)
        
        encoding = sys.getfilesystemencoding()
        headers = [("Content-type", "application/json; charset=%s" % encoding),
                   ("Content-Length", str(len(js_dir)))]
        
        return ResponseOK(js_dir, headers=headers)

#    
#    def log_request(self, code='-', size='-'):
#        """Log an accepted request.
#
#        This is called by send_response().
#
#        """
#        if self.logrequests:
#            self.log_message('"%s" %s %s', self.requestline, str(code), str(size))
#
#    def log_error(self, format, *args):
#        """Log an error.
#
#        This is called when a request cannot be fulfilled.  By
#        default it passes the message on to log_message().
#
#        Arguments are the same as for log_message().
#
#        XXX This should go to the separate error log.
#
#        """
#        logger.error("%s - - [%s] %s\n" %
#                         (self.client_address[0],
#                          self.log_date_time_string(),
#                          format % args))
#
#        #self.log_message(format, *args)
#
#    def log_message(self, format, *args):
#        """Log an arbitrary message.
#
#        This is used by all other logging functions.  Override
#        it if you have specific logging wishes.
#
#        The first argument, FORMAT, is a format string for the
#        message to be logged.  If the format string contains
#        any % escapes requiring parameters, they should be
#        specified as subsequent arguments (it's just like
#        printf!).
#
#        The client ip address and current date/time are prefixed to every
#        message.
#
#        """
#
#        logger.info("%s - - [%s] %s\n" %
#                         (self.client_address[0],
#                          self.log_date_time_string(),
#                          format % args))
#
#        
#
#    def send_error(self, code, message=None):
#        """Send and log an error reply.
#
#        Arguments are the error code, and a detailed message.
#        The detailed message defaults to the short entry matching the
#        response code.
#
#        This sends an error response (so it must be called before any
#        output has been generated), logs the error, and finally sends
#        a piece of HTML explaining the error to the user.
#
#        """
#
#        try:
#            short, long = self.responses[code]
#        except KeyError:
#            short, long = '???', '???'
#        if message is None:
#            message = short
#        explain = long
#        self.log_error("code %d, message %s", code, message)
#        self.send_response(code, message)
#        #self.send_header('Connection', 'close')
#
#        # Message body is omitted for cases described in:
#        #  - RFC7230: 3.3. 1xx, 204(No Content), 304(Not Modified)
#        #  - RFC7231: 6.3.6. 205(Reset Content)
#        content = None
#        if code >= 200 and code not in (204, 205, 304):
#            # HTML encode to prevent Cross Site Scripting attacks
#            # (see bug #1100201)
#            content = (self.error_message_format % {
#                'code': code,
#                'message': BaseHTTPServer._quote_html(message),
#                'explain': explain
#            })
#            self.send_header("Content-Type", self.error_content_type)
#            self.send_header("Content-Length", str(len(content)))
#
#        self.end_headers()
#
#        if self.command != 'HEAD' and content:
#            self.wfile.write(content)




def main(protocol="HTTP/1.0"):
    """Test the HTTP request handler class.

    This runs an HTTP server on port 8000 (or the first command line
    argument).

    """
    from optparse import OptionParser


    op = OptionParser(usage='usage: %s [options] [filename]' % sys.argv[0])

    op.add_option('-p', '--port', dest='port', default=config.get('dataserver-port', 8080),
                  help="port number to serve on")
    op.add_option('-t', '--test', dest='test', help="Set up for bandwidth test (don't save files)", action="store_true", default=False)
    op.add_option('-v', '--protocol', dest='protocol', help="HTTP protocol version", default="1.1")
    op.add_option('-l', '--log-requests', dest='log_requests', help="Display http request info", default=False, action="store_true")
    op.add_option('-r', '--root', dest='root', help="Root directory of virtual filesystem", default=config.get('dataserver-root', os.curdir))


    options, args = op.parse_args()
    
    #change to the dataserver root if given
    logger.info('Serving from directory: %s' % options.root)
    os.chdir(options.root)



    #PYMEHTTPRequestHandler.protocol_version = 'HTTP/%s' % options.protocol
    server_config['bandwidthTesting'] = options.test
    #PYMEHTTPRequestHandler.logrequests = options.log_requests

    #httpd = ThreadedHTTPServer(server_address, PYMEHTTPRequestHandler)
    ip_addr = socket.gethostbyname(socket.gethostname())
    server_address = ('', int(options.port))

    global_status['IPAddress'] = ip_addr
    global_status['BindAddress'] = server_address
    global_status['Port'] = int(options.port)
    global_status['Protocol'] = options.protocol
    global_status['TestMode'] = options.test
    global_status['ComputerName'] = GetComputerName()

    ns = pzc.getNS('_pyme-http')
    service_name = get_service_name('PYMEDataServer')
    ns.register_service(service_name, ip_addr, int(options.port))

    print("Serving HTTP on %s port %d ..." % (ip_addr, options.port))

    #wsgiref_server(options)
    cherrypy_server(options)

def wsgiref_server(options):
    ds = dataserver()
    httpd = make_server('', int(options.port), ds)

    try:
        httpd.serve_forever()
    finally:
        logger.info('Shutting down ...')
        httpd.shutdown()
        httpd.server_close()
        sys.exit()

def cherrypy_server(options):
    from cherrypy import wsgiserver
    ds = dataserver()

    server = wsgiserver.CherryPyWSGIServer(('0.0.0.0', int(options.port)), ds)

    try:
        server.start()
    finally:
        logger.info('Shutting down ...')
        server.stop()
        sys.exit()

if __name__ == '__main__':
    main()
