import BaseHTTPServer
from SocketServer import ThreadingMixIn
import urlparse

def register_endpoint(path):
    def _reg_ep(func):
        #_endpoints[path] = func
        func._expose_path = path
        return func

    return _reg_ep

class JSONAPIRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'

    def _process_request(self):
        up = urlparse.urlparse(self.path)

        kwargs = urlparse.parse_qs(up.query)

        cl = int(self.headers.get('Content-Length', 0))
        if cl > 0:
            body = self.rfile.read(cl)
            kwargs['body'] = body

        handler = self.server._endpoints[up.path]
        resp = handler(**kwargs)

        self.send_response(200)
        self.send_header("Content-Type", 'application/json')
        self.send_header("Content-Length", "%d" % len(resp))
        self.end_headers()

        self.wfile.write(resp)
        return


    def do_GET(self):
        return self._process_request()

    def do_POST(self):
        return self._process_request()


class APIHTTPServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
    def __init__(self, server_address):
        BaseHTTPServer.HTTPServer.__init__(self, server_address, JSONAPIRequestHandler)

        #make a mapping of endpoints to functions
        self._endpoints = {}
        for a in dir(self):
            func = getattr(self, a)
            endpoint_path = getattr(func, '_expose_path', None)
            if not endpoint_path is None:
                self._endpoints[endpoint_path] = func