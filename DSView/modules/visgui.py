#!/usr/bin/python
##################
# visgui.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################


from PYME.Analysis.LMVis.imageView import *

def Plug(dsviewer):
    cmaps = [pylab.cm.r, pylab.cm.g, pylab.cm.b]

    if not ivps in dir('dsviewer'):
        dsviewer.ivps = []

    for name, i in zip(dsviewer.image.names, xrange(dsviewer.image.data.shape[3])):
        dsviewer.ivps.append(ImageViewPanel(dsviewer, dsviewer.image, dsviewer.glCanvas, dsviewer.do, chan=i))
        if dsviewer.image.data.shape[3] > 1 and len(cmaps) > 0:
            dsviewer.do.cmaps[i] = cmaps.pop(0)

        dsviewer.AddPage(page=dsviewer.ivps[-1], select=True, caption=name)


    if dsviewer.image.data.shape[2] > 1:
        dsviewer.AddPage(page=ArrayViewPanel(dsviewer, do=dsviewer.do, aspect = asp), select=False, caption='Slices')

    elif dsviewer.image.data.shape[3] > 1:
        dsviewer.civp = ColourImageViewPanel(dsviewer, dsviewer.glCanvas, dsviewer.do, dsviewer.image)
        dsviewer.civp.ivps = dsviewer.ivps
        dsviewer.AddPage(page=dsviewer.civp, select=True, caption='Composite')
    

