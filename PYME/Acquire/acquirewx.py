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
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from PYME.Acquire.acquiremainframe import PYMEMainFrame   
from PYME.Acquire.acquire_server import AcquireHTTPServerMixin

def create(parent, options = None):
    return AcquireWxHTTPServer(parent, options, port=int(options.port), bind_addr=options.bind_addr)

class AcquireWxHTTPServer(PYMEMainFrame, AcquireHTTPServerMixin):
    """Server with wx GUI"""
    def __init__(self, parent, options, port, bind_addr=None):
        assert(options.threaded_event_loop)
        
        PYMEMainFrame.__init__(self, parent, options)
        AcquireHTTPServerMixin.__init__(self, port, bind_addr)

    def run(self):
        import threading
        PYMEMainFrame.run(self)

        if self.options.server:
            # only start the server if requested
            self._server_thread = threading.Thread(target=self.serve_forever)
            self._server_thread.start()

            if self.options.ipy:
                # make this a separate config option as port is hard coded so can't run more than one
                # process with this option. Also probably not desirable if you just want progamatic
                # remote control (through REST API).
                from PYME.Acquire.webui import ipy
                ns = dict(scope=self.scope, server=self)
                print('namespace:', ns)
                ipy.launch_ipy_server_thread(user_ns=ns)
