#!/usr/bin/python

##################
# PYMEAquire.py
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
This is the principle entry point for `PYMEAcquire`, the acquisition component of PYME.

`PYMEAcquire` takes one option to specify the initialisation file, which should be in the 
'PYME/Acquire/Scripts' directory

.. code-block:: bash

    python PYMEAcquire.py -i <initialisation file>
    
If run without an intialisation file it defaults to using simulated hardware.
"""
from PYME.misc import big_sur_fix
from PYME.ui import patch_traitsui

#!/usr/bin/python
import wx
#import matplotlib
#matplotlib.use('WXAgg')
#from PYME import mProfile

#make wx less spammy with warnings
import warnings
warnings.simplefilter('once', wx.wxPyDeprecationWarning)

import os
import logging
import logging.config

def setup_logging(default_level=logging.DEBUG):
    """Setup logging configuration

    """
    import yaml
    import PYME.config
    from PYME.util import log_verbosity
    default_config_file = os.path.join(os.path.split(__file__)[0], 'logging.yaml')
    
    path = PYME.config.get('Acquire-logging_conf_file', default_config_file)
    print('attempting to load load logging config from %s' % path)
    
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    # suppress excessively verbose logging in dependency packages
    log_verbosity.patch_log_verbosity()

class BoaApp(wx.App):
    def __init__(self, options, *args):
        self.options = options
        wx.App.__init__(self, *args)
        
        
    def OnInit(self):
        if self.options.server:
            from PYME.Acquire import acquirewx as acquiremainframe
        else:
            from PYME.Acquire import acquiremainframe

        self.main = acquiremainframe.create(None, self.options)
        self.SetTopWindow(self.main)

        if self.options.browser:
            import webbrowser
            webbrowser.open('http://localhost:8999') #FIXME - delay this until server is up
        return True


def main():
    import os
    import sys
    #from optparse import OptionParser
    import argparse
    setup_logging()
    
    from PYME import config
    
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init-file", dest="initFile",
                      help="Read initialisation from file [defaults to init.py]",
                      metavar="FILE", default='init.py')

    parser.add_argument("-m", "--gui_mode", dest="gui_mode", default='default',
                      help="GUI mode for PYMEAcquire - either default or 'compact'")

    parser.add_argument("-t", "--title", dest="window_title", default='PYME Acquire',
                      help="Set the PYMEAcquire display name (useful when running multiple copies - e.g. for drift tracking)")
    
    parser.add_argument('-p', '--port', dest='port', default=8999, help='port to use for server functions')
    parser.add_argument('-a', '--bind_addr', dest='bind_addr', default=None, help='address to bind to for server functions (defaults to localhost). Only bind to an external address if you are on a trusted network and *really* know what you are doing. NB - university networks should generally not be trusted.')
    parser.add_argument('-s', '--server', dest='server', default=False, action='store_true', help='run in server mode')
    parser.add_argument('-b', '--browser', dest='browser', default=False, action='store_true', help='launch web browser based ui')
    parser.add_argument('-e', '--threaded_event_loop', dest='threaded_event_loop', default=False, action='store_true', help='Run hardware event loop in separate thread. Required (and implied) for server mode.')
    parser.add_argument('-I', '--ipy', dest='ipy', default=False, action='store_true', help='launch ipython server for remote control')
    parser.add_argument('--no-wx', dest='no_wx', default=False, action='store_true', help='run without wx gui')

    #(options, args) = parser.parse_args()

    options = parser.parse_args()

    print(options)

    if options.server:
        options.threaded_event_loop = True

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

    logger.debug('using initialization script %s, %s' % (init_file, os.path.abspath(init_file)))

    if options.no_wx:
        #implies server mode (with or without a browser GUI)
        assert options.server

        from PYME.Acquire.acquire_server import AcquireHTTPServer
        server = AcquireHTTPServer(options, port=int(options.port), bind_addr=options.bind_addr)
        if options.browser:
            import webbrowser
            webbrowser.open('http://localhost:8999') #FIXME - delay this until server is up

        server.run()
    else:
        application = BoaApp(options, 0)
        application.MainLoop()

if __name__ == '__main__':
    from PYME.util import mProfile, fProfile
    #mProfile.profileOn(['acquiremainframe.py', 'microscope.py', 'frameWrangler.py', 'fakeCam.py', 'rend_im.py'])
    #fp = fProfile.thread_profiler()
    #fp.profileOn()
    main()
    #fp.profileOff()
    #mProfile.report()
