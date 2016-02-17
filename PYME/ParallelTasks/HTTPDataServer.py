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

import SimpleHTTPServer
import BaseHTTPServer
import os
from StringIO import StringIO
import shutil
#import urllib
import sys
import json
import PYME.misc.pyme_zeroconf as pzc

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()
procName = compName + ' - PID:%d' % os.getpid()

class PYMEHTTPRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_PUT(self):
        path = self.translate_path(self.path)
        
        #print self.headers
        
        if os.path.exists(path):
            #Do not overwrite - we use write-once semantics
            self.send_error(405, "File already exists")
            return None
        else:
            dir = os.path.split(path)[0]
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(path, 'wb') as f:
                #shutil.copyfileobj(self.rfile, f, int(self.headers['Content-Length']))
                f.write(self.rfile.read(int(self.headers['Content-Length'])))
            
            self.send_response(200)
            return
            
    def list_directory(self, path):
        """Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().

        """
        try:
            list = os.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None
        list.sort(key=lambda a: a.lower())
        f = StringIO()
        #displaypath = cgi.escape(urllib.unquote(self.path))
        l2 = []
        for l in list:
            if os.path.isdir(l):
                l2.append(l + '/')
            else:
                l2.append(l)
        f.write(json.dumps(l2))
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        encoding = sys.getfilesystemencoding()
        self.send_header("Content-type", "application/json; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f
        
    
def test(protocol="HTTP/1.0"):
    """Test the HTTP request handler class.

    This runs an HTTP server on port 8000 (or the first command line
    argument).

    """

    if sys.argv[1:]:
        port = int(sys.argv[1])
    else:
        port = 8000
    server_address = ('', port)

    PYMEHTTPRequestHandler.protocol_version = protocol
    httpd = BaseHTTPServer.HTTPServer(server_address, PYMEHTTPRequestHandler)

    sa = httpd.socket.getsockname()

    ns = pzc.getNS('_pyme-http')
    ns.register_service('PYMEDataServer: ' + procName, sa[0], sa[1])    
    
    print "Serving HTTP on", sa[0], "port", sa[1], "..."
    httpd.serve_forever()


if __name__ == '__main__':
    test()