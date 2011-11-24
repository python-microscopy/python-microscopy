#!/usr/bin/python
from PYME.Analysis.LMVis.VisGUI import *


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    filename = None

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    if wx.GetApp() == None: #check to see if there's already a wxApp instance (running from ipython -pylab or -wthread)
        main(filename)
    else:
        #time.sleep(1)
        visFr = VisGUIFrame(None, filename)
        visFr.Show()
        visFr.RefreshView()
