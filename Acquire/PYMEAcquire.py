#!/usr/bin/python

##################
# PYMEAquire.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
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
    mProfile.profileOn(['displaySettingsPanel.py']) #'AndorNeo.py', 'previewaquisator.py', 
    main()
    mProfile.report()
