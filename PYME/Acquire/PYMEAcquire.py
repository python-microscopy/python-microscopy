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

#!/usr/bin/python
import wx
from PYME.Acquire import acquiremainframe
#from PYME import mProfile

import os
import json
import logging
import logging.config

def setup_logging(
    default_path='logging.json', 
    default_level=logging.DEBUG,
    env_key='LOG_CFG'
    ):
    """Setup logging configuration

    """
    path = os.path.join(os.path.split(__file__)[0], default_path)
    print path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


class BoaApp(wx.App):
    def __init__(self, options, *args):
        self.options = options
        wx.App.__init__(self, *args)
        
        
    def OnInit(self):
        #wx.InitAllImageHandlers()
        self.main = acquiremainframe.create(None, self.options)
        #self.main.Show()
        self.SetTopWindow(self.main)
        return True


def main():
    from optparse import OptionParser
    setup_logging()

    parser = OptionParser()
    parser.add_option("-i", "--init-file", dest="initFile", help="Read initialisation from file [defaults to init.py]", metavar="FILE")
        
    (options, args) = parser.parse_args()
    
    application = BoaApp(options, 0)
    application.MainLoop()

if __name__ == '__main__':
    from PYME.util import mProfile
    mProfile.profileOn(['acquiremainframe.py', 'microscope.py', 'frameWrangler.py', 'fakeCam.py', 'rend_im.py'])
    main()
    mProfile.report()
