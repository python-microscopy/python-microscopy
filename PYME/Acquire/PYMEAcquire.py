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

#!/usr/bin/python
import wx
import acquiremainframe
#from PYME import mProfile


class BoaApp(wx.App):
    def __init__(self, options, *args):
        self.options = options
        wx.App.__init__(self, *args)
        
        
    def OnInit(self):
        wx.InitAllImageHandlers()
        self.main = acquiremainframe.create(None, self.options)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True


def main():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--init-file", dest="initFile", help="Read initialisation from file [defaults to init.py]", metavar="FILE")
        
    (options, args) = parser.parse_args()
    
    application = BoaApp(options, 0)
    application.MainLoop()

if __name__ == '__main__':
    from PYME import mProfile
    mProfile.profileOn(['rend_im.py'])
    main()
    mProfile.report()
