#!/usr/bin/python

##################
# smimainframe.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################
"""
This contains the bulk of the GUI code for the main window of PYMEAcquire.
"""
#import matplotlib #import before we start logging to avoid log messages in debug

import os

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from PYME.util import webframework
import threading

#from PYME.Acquire import webui

from PYME.Acquire.acquirebase import PYMEAcquireBase

class PYMEAcquireServerMixin(object):
    def __init__(self, *args, **kwargs):
       pass
        
    @webframework.register_endpoint('/get_frame_pzf', mimetype='image/pzf')
    def get_frame_pzf(self):
        """
        Get a frame in PZF format (compressed, fast), uses long polling
        
        Returns
        -------

        """
        from PYME.IO import PZFFormat
        with self._new_frame_condition:
            while self._current_frame is None:
                self._new_frame_condition.wait()
                #logger.debug(self._current_frame is None)
                
            ret = PZFFormat.dumps(self._current_frame, compression=PZFFormat.DATA_COMP_RAW)
            self._current_frame = None
            
        return ret
    
    @webframework.register_endpoint('/get_frame_png', mimetype='image/png')
    def get_frame_png(self, min=None, max=None):
        """
        Get a frame in PNG format
        
        uses long polling
        
        Returns
        -------

        """
        import numpy as np
        from io import BytesIO
        from PIL import Image

        out = BytesIO()
        
        with self._new_frame_condition:
            while self._current_frame is None:
                self._new_frame_condition.wait()

            #im = np.sqrt(self._current_frame.squeeze()).astype('uint8')
            
            im = self._current_frame.squeeze()
                 
            if min is None:
                min = im.min()
            else:
                min=float(min)
            
            if max is None:
                max = im.max()
            else:
                max= float(max)
                
                
            im = (255*(im - min)/(max-min)).astype('uint8')

            Image.fromarray(im.T).save(out, 'PNG')
            self._current_frame = None
            
        s = out.getvalue()
        out.close()
        return s

    @webframework.register_endpoint('/get_frame_png_b64', mimetype='image/png')
    def get_frame_png_b64(self, min=None, max=None):
        import base64
        return base64.b64encode(self.get_frame_png(min, max))

    @webframework.register_endpoint('/get_frame_raw', mimetype='application/octet-stream')
    def get_frame_raw(self, min=None, max=None):
        import numpy as np
        with self._new_frame_condition:
            while self._current_frame is None:
                self._new_frame_condition.wait()
                #logger.debug(self._current_frame is None)
        
            ret = bytes(self._current_frame.data)
            #ret=bytes(np.arange(5).data)
            self._current_frame = None
    
        return ret
        
    @webframework.register_endpoint('/get_scope_state', output_is_json=False)
    def get_scope_state(self, keys=None):
        """
        Gets the current scope state as a json dictionary
        
        Parameters
        ----------
        keys : list, optional
          a list of keys to interrogate. If none, returns full state.

        Returns
        -------

        """
        
        if keys is None:
            keys = self.scope.state.keys()
            
        return {k : self.scope.state[k] for k in keys}

    @webframework.register_endpoint('/scope_state_longpoll', output_is_json=False)
    def scope_state_longpoll(self, keys=None):
        """
        Gets the current scope state as a json dictionary, only returning once the state has changed

        Parameters
        ----------
        keys : list, optional
          a list of keys to interrogate. If none, returns full state.

        Returns
        -------

        """
        if keys is None:
            keys = self.scope.state.keys()
            
        with self._state_updated_condition:
            while self._state_valid:
                self._state_updated_condition.wait()
    
            logger.debug('returning updated state')
            ret = {k: self.scope.state[k] for k in keys}
            self._state_valid = True
    
        return ret
    
    @webframework.register_endpoint('/update_scope_state', output_is_json=False)
    def update_scope_state(self, body=''):
        import json
        state = json.loads(body)
        
        self.scope.state.update(state)
        
        return 'OK' #TODO - check for errors


from PYME.Acquire import webui
from PYME.Acquire import SpoolController
class AcquireHTTPServerMixin(webframework.APIHTTPServer, PYMEAcquireServerMixin):
    def __init__(self, port, bind_addr=None):
        PYMEAcquireServerMixin.__init__(self)

        if bind_addr is None:
            bind_addr = 'localhost' # bind to localhost by default in an attempt to make this safer
        
        server_address = (bind_addr, port)
        webframework.APIHTTPServer.__init__(self, server_address)
        self.daemon_threads = True
        
        self.add_endpoints(SpoolController.SpoolControllerWrapper(self.scope.spoolController), '/spool_controller')
        #self.add_endpoints(self.scope.stackSettings, '/stack_settings')
        self.add_static_handler('static', webframework.StaticFileHandler(os.path.join(os.path.dirname(__file__), 'webui', 'static')))
        
        webui.set_server(self)
        
        self._main_page = webui.load_template('PYMEAcquire.html')
        
    @webframework.register_endpoint('/do_login')
    def do_login(self, email, password, on_success='/'):
        from PYME.util import authenticate
        
        try:
            auth = authenticate.get_token(email, password)
        except:
            #logger.exception('Error getting auth token')
            auth = None
            
        if auth:
            return webframework.HTTPRedirectResponse(on_success, headers=[('Set-Cookie', 'auth=%s; path=/; HttpOnly' % auth)])
        else:
            return webframework.HTTPRedirectResponse('/login?reason="failure"&on_success="%s"'%on_success, headers=[('Set-Cookie', 'auth=; path=/; HttpOnly; expires=Thu, 01 Jan 1970 00:00:00 GMT')])
     
    @webframework.register_endpoint('/login', mimetype='text/html')
    def login(self, reason='', on_success='/'):
        from jinja2 import Template
        
        return Template(webui.load_template('login.html')).render(reason=reason, on_success=on_success)

    @webframework.register_endpoint('/logout')
    def logout(self, on_success='/'):
        return webframework.HTTPRedirectResponse(on_success, headers=[
            ('Set-Cookie', 'auth=; path=/; HttpOnly; expires=Thu, 01 Jan 1970 00:00:00 GMT')])

    @webframework.register_endpoint('/', mimetype='text/html', authenticate=True)
    def main_page(self, authenticated_as=None):
        #return self._main_page
        from jinja2 import Template
        
        print('authenticated_as=', authenticated_as)
        return Template(webui.load_template('PYMEAcquire.html')).render(authenticated_as=authenticated_as)
        
    def run(self):
        self._poll_thread = threading.Thread(target=self.main_loop)
        self._poll_thread.start()
        
        try:
            self.serve_forever()
        finally:
            self.evt_loop.stop()
            #logger.info('Shutting down ...')
            #self.distributor.shutdown()
            logger.info('Closing server ...')
            self.server_close()

class AcquireHTTPServer(PYMEAcquireBase, AcquireHTTPServerMixin):
    """Server without wx GUI"""
    def __init__(self, options, port, bind_addr=None, evt_loop=None):
        PYMEAcquireBase.__init__(self, options, evt_loop=evt_loop)
        AcquireHTTPServerMixin.__init__(self, port, bind_addr)


def main():
    import os
    import sys
    from optparse import OptionParser
    logging.basicConfig(level=logging.DEBUG)
    
    from PYME import config
    from PYME.Acquire.webui import ipy
    
    logger = logging.getLogger(__name__)
    parser = OptionParser()
    parser.add_option("-i", "--init-file", dest="initFile",
                      help="Read initialisation from file [defaults to init.py]",
                      metavar="FILE", default='init.py')
    parser.add_option('-b', '--reuse-browser', dest="browser", default=True, action="store_false")
    
    (options, args) = parser.parse_args()
    
    # continue to support loading scripts from the PYMEAcquire/Scripts directory
    legacy_scripts_dir = os.path.join(os.path.dirname(__file__), 'Scripts')
    
    # use new config module to locate the initialization file
    init_file = config.get_init_filename(options.initFile, legacy_scripts_directory=legacy_scripts_dir)
    if init_file is None:
        logger.critical('init script %s not found - aborting' % options.initFile)
        sys.exit(1)
    
    #overwrite initFile in options with full path - CHECKME - does this work?
    options.initFile = init_file
    logger.info('using initialization script %s' % init_file)
    
    server = AcquireHTTPServer(options, 8999)
    ns = dict(scope=server.scope, server=server)
    print('namespace:', ns)
    ipy.launch_ipy_server_thread(user_ns=ns)
    
    if options.browser:
        import webbrowser
        webbrowser.open('http://localhost:8999') #FIXME - delay this until server is up
    
    server.run()
    


if __name__ == '__main__':
    from PYME.util import mProfile, fProfile
    
    #mProfile.profileOn(['acquire_server.py', 'microscope.py', 'frameWrangler.py'])
    #fp = fProfile.thread_profiler()
    #fp.profileOn()
    try:
        main()
    finally:
        #fp.profileOff()
        #mProfile.report()
        pass