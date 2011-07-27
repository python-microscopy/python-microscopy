#!/usr/bin/python
##################
# arrayView.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.DSView.arrayViewPanel import *

def Plug(dsviewer):
    dsviewer.view = ArrayViewPanel(dsviewer, do=dsviewer.do)
    dsviewer.AddPage(dsviewer.view, True, 'Data')

#    dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
#    dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
#    pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
#    dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)

#    dsviewer._mgr.AddPane(dsviewer.view.CreateToolBar(dsviewer), aui.AuiPaneInfo().Name("ViewTools").Caption("View Tools").CloseButton(False).
#                      ToolbarPane().Right().GripperTop())

    #dsviewer.panesToMinimise.append(pinfo2)


