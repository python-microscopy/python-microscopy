# noinspection PyCompatibility
import http.server

# noinspection PyCompatibility
from socketserver import ThreadingMixIn

try:
    # noinspection PyCompatibility
    import urlparse
    import Cookie as cookies
except ImportError:
    #py3
    # noinspection PyCompatibility
    import urllib.parse as urlparse
    from http import cookies

import json
    
import logging
logger = logging.getLogger(__name__)

WEBSOCKET_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def register_endpoint(path, output_is_json=True, mimetype='application/json', compress=True, cookies=False, authenticate=False):
    def _reg_ep(func):
        #_endpoints[path] = func
        func._expose_path = path
        func._jsonify = not output_is_json
        func._mimetype = mimetype
        func._compress = compress
        func._parse_cookies = cookies
        func._authenticate = authenticate
            
        return func

    return _reg_ep

class HTTPResponse(object):
    """ permit setting headers etc"""
    def __init__(self, body, headers=None, response_code=200):
        self.body = body
        self.headers= headers
        self.response_code = response_code
        
    def write_response(self, method, handler):
        resp = self.body
        if method._jsonify:
            resp = json.dumps(resp)
    
        compress_output = method._compress and ('gzip' in handler.headers.get('Accept-Encoding', ''))
    
        handler.send_response(self.response_code)
        handler.send_header("Content-Type", method._mimetype)
        if compress_output:
            handler.send_header('Content-Encoding', 'gzip')
            resp = handler._gzip_compress(resp) #FIXME - write directly to wfile rather than to a BytesIO object - how do we find the content-length?
    
        handler.send_header("Content-Length", "%d" % len(resp))
        if self.headers is not None:
            for k, v in self.headers:
                handler.send_header(k, v)
        handler.end_headers()
    
        handler.wfile.write(resp)
        
class HTTPRedirectResponse(HTTPResponse):
    def __init__(self, redirect_to, headers=None, response_code=303):
        new_headers = [('Location', redirect_to)]
        if not headers is None:
            new_headers.extend(headers)
        HTTPResponse.__init__(self, '', new_headers, response_code)
        
        

class JSONAPIRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    logrequests = False

    def _gzip_compress(self, data):
        import gzip
        from io import BytesIO
        
        if not isinstance(data, bytes):
            data = data.encode()
        
        zbuf = BytesIO()
        zfile = gzip.GzipFile(mode='wb', fileobj=zbuf, compresslevel=3)
        zfile.write(data)
        zfile.close()
    
        return zbuf.getvalue()

    def _gzip_decompress(self, data):
        import gzip
        from io import BytesIO
        zbuf = BytesIO(data)
        zfile = gzip.GzipFile(mode='rb', fileobj=zbuf)#, compresslevel=9)
        out = zfile.read()
        zfile.close()
    
        return out

    def _process_request(self):
        #import gzip
        up = urlparse.urlparse(self.path)

        kwargs = urlparse.parse_qs(up.query)

        kwargs = {k : v[0] for k, v in kwargs.items()}

        cl = int(self.headers.get('Content-Length', 0))
        if cl > 0:
            body = self.rfile.read(cl)
            
            if self.headers.get('Content-Encoding') == 'gzip':
                body = self._gzip_decompress(body)
                
            kwargs['body'] = body

        #logger.debug('Request path: ' + up.path)
        #logger.debug('Requests args: ' + repr(kwargs))

        try:
            handler = self.server._endpoints[up.path]
        except KeyError:
            self.send_error(404, 'No handler for %s' % up.path)
            return
        
        try:
            kwargs.pop('authenticated_as') #if anything fails we are not authenticated NB - this stops people from passing authenticated_as on the query string.
        except KeyError:
            pass
        
        if handler._parse_cookies or  handler._authenticate:
            req_cookies = cookies.SimpleCookie(self.headers.get('Cookie'))
            if handler._parse_cookies:
                kwargs['cookies'] = req_cookies
                
            if handler._authenticate:
                from PYME.util import authenticate
                try:
                    auth_token = req_cookies.get('auth').value
                    kwargs['authenticated_as'] = authenticate.validate_token(auth_token)['email']
                except:
                    pass
                    
         
        if self.headers.get('Upgrade', None) == 'websocket':
            self._websocket_upgrade(handler, kwargs)
            return
        
        try:
            resp = handler(**kwargs)
        except Exception:
            import traceback
            self.send_error(500, 'Server Error\n %s' % traceback.format_exc())
            return
        
        if isinstance(resp, HTTPResponse):
            resp.write_response(handler, self)
            return
        
        if handler._jsonify:
            resp = json.dumps(resp)
        
        compress_output = handler._compress and ('gzip' in self.headers.get('Accept-Encoding', ''))

        self.send_response(200)
        self.send_header("Content-Type", handler._mimetype)
        if compress_output:
            self.send_header('Content-Encoding', 'gzip')
            resp = self._gzip_compress(resp) #FIXME - write directly to wfile rather than to a BytesIO object - how do we find the content-length?
            
        self.send_header("Content-Length", "%d" % len(resp))
        self.end_headers()

        self.wfile.write(resp)
        return
    
    
    def _websocket_upgrade(self, handler, kwargs):
        ws_key = self.headers.get('Sec-WebSocket-Key')
        ws_version = self.headers.get('Sec-WebSocket-Version', 0)

        self.send_response(101)
        
        
        


    def do_GET(self):
        return self._process_request()

    def do_POST(self):
        return self._process_request()

    def log_request(self, code='-', size='-'):
        """Log an accepted request.

        This is called by send_response().

        """
        if self.logrequests:
            self.log_message('"%s" %s %s', self.requestline, str(code), str(size))


class APIHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    def __init__(self, server_address):
        http.server.HTTPServer.__init__(self, server_address, JSONAPIRequestHandler)

        #make a mapping of endpoints to functions
        self._endpoints = {}
        for a in dir(self):
            func = getattr(self, a)
            endpoint_path = getattr(func, '_expose_path', None)
            if not endpoint_path is None:
                self._endpoints[endpoint_path] = func
                
        logging.debug('Registered endpoints: %s' % self._endpoints.keys())