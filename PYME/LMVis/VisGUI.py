#!/usr/bin/python
##################
# VisGUI.py
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
import argparse

import wx
import wx.py.shell

#import PYME.ui.autoFoldPanel as afp
#import wx.lib.agw.aui as aui

#hacked so py2exe works
#from PYME.DSView.dsviewer import View3D

#from PYME.LMVis import gl_render
#from PYME.LMVis import workspaceTree
import sys

import matplotlib
matplotlib.use('wxagg')
import pylab

from PYME.misc import extraCMaps
from PYME.IO.FileUtils import nameUtils

#import os
#from PYME.LMVis import gl_render3D

from PYME.LMVis import colourPanel
#from PYME.LMVis import renderers
from PYME.LMVis import pipeline

import logging
logger = logging.getLogger(__name__)


from PYME.ui import MetadataTree

import numpy as np

from PYME.DSView import eventLogViewer

from PYME.LMVis import statusLog
from PYME.LMVis import visCore

from PYME.ui.AUIFrame import AUIFrame
####################################        
#defines the main GUI class fo VisGUI

class VisGUIFrame(AUIFrame, visCore.VisGUICore):
    """The main GUI frame for VisGUI. Note that much of the functionality is shared
    with the LMDisplay module used for online display and has been factored out into the visCore module"""
    def __init__(self, parent, filename=None, id=wx.ID_ANY, 
                 title="PYME Visualise", pos=wx.DefaultPosition,
                 size=(700,650), style=wx.DEFAULT_FRAME_STYLE, use_shaders=False):

        AUIFrame.__init__(self, parent, id, title, pos, size, style)
        

        self._flags = 0
        
        self.pipeline = pipeline.Pipeline(visFr=self)
        
        #self.Quads = None
               
        #self.SetMenuBar(self.CreateMenuBar())
        self.CreateMenuBar(use_shaders=use_shaders)

        self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)

        self.statusbar.SetStatusText("", 0)
       
        #self._leftWindow1 = wx.Panel(self, -1, size = wx.Size(220, 1000))
        #self._pnl = 0
        
        #initialize the common parts
        ###############################
        #NB: this has to come after the shell has been generated, but before the fold panel
        visCore.VisGUICore.__init__(self, use_shaders=use_shaders)

        ################################   

        self.MainWindow = self #so we can access from shell
        self.sh = wx.py.shell.Shell(id=-1,
              parent=self, size=wx.Size(-1, -1), style=0, locals=self.__dict__,
              introText='Python SMI bindings - note that help, license etc below is for Python, not PySMI\n\n')

        #self._mgr.AddPane(self.sh, aui.AuiPaneInfo().
        #                  Name("Shell").Caption("Console").Centre().CloseButton(False).CaptionVisible(False))

        self.AddPage(self.sh, caption='Shell')
             
        
        self.elv = None
        self.colp = None
        self.mdp = None
        self.rav = None

        self.generatedImages = []
        
#        if 'PYME_BUGGYOPENGL' in os.environ.keys():
#            pylab.plot(pylab.randn(10))

        self.sh.Execute('from pylab import *')
        self.sh.Execute('from PYME.DSView.dsviewer import View3D')

        #self.workspace = workspaceTree.WorkWrap(self.__dict__)
        ##### Make certain things visible in the workspace tree

        #components of the pipeline
        #col = self.workspace.newColour()
        #self.workspace.addKey('pipeline', col)
        
        #Generated stuff
        #col = self.workspace.newColour()
        #self.workspace.addKey('GeneratedMeasures', col)
        #self.workspace.addKey('generatedImages', col)
        #self.workspace.addKey('objects', col)

        #main window, so we can get everything else if needed
        #col = self.workspace.newColour()
        #self.workspace.addKey('MainWindow', col)

        ######

        #self.workspaceView = workspaceTree.WorkspaceTree(self, workspace=self.workspace, shell=self.sh)
        #self.AddPage(page=wx.StaticText(self, -1, 'foo'), select=False, caption='Workspace')

#        self.glCanvas = gl_render.LMGLCanvas(self)
#        self.AddPage(page=self.glCanvas, select=True, caption='View')
#        self.glCanvas.cmap = pylab.cm.gist_rainbow #pylab.cm.hot

        #self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_MOVE, self.OnMove)
        self.Bind(wx.EVT_CLOSE, self.OnQuit)

        #self.Bind(wx.EVT_IDLE, self.OnIdle)
        #self.refv = False

        statusLog.SetStatusDispFcn(self.SetStatus)
        
        
        self.paneHooks.append(self.GenPanels)
        self.CreateFoldPanel()


        if not filename is None:
            self.OpenFile(filename)
            #self.refv = False
            wx.CallAfter(self.RefreshView)

        nb = self._mgr.GetNotebooks()[0]
        nb.SetSelection(0)
        

    def OnMove(self, event):
        self.Refresh()
        self.Update()
        event.Skip()      

    def OnQuit(self, event):
        while len(self.pipeline.filesToClose) > 0:
            self.pipeline.filesToClose.pop().close()

        pylab.close('all')
        self._cleanup()


    def OnAbout(self, event):
        msg = "PYME Visualise\n\n Visualisation of localisation microscopy data\nDavid Baddeley 2009"
              
        dlg = wx.MessageDialog(self, msg, "About PYME Visualise",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.SetFont(wx.Font(8, wx.NORMAL, wx.NORMAL, wx.NORMAL, False, "Verdana"))
        dlg.ShowModal()
        dlg.Destroy()

#    def OnToggleWindow(self, event):
#        self._mgr.ShowPane(self._leftWindow1,not self._leftWindow1.IsShown())
#        self.glCanvas.Refresh()  
            

    # def OnView3DPoints(self,event):
    #     if 'z' in self.pipeline.keys():
    #         if not 'glCanvas3D' in dir(self):
    #             #self.glCanvas3D = gl_render3D.LMGLCanvas(self)
    #             #self.AddPage(page=self.glCanvas3D, select=True, caption='3D')
    #             self.glCanvas3D = gl_render3D.showGLFrame()

    #         #else:            
    #         self.glCanvas3D.setPoints3D(self.pipeline['x'], 
    #                               self.pipeline['y'], 
    #                               self.pipeline['z'], 
    #                               self.pointColour())
    #         self.glCanvas3D.setCLim(self.glCanvas.clim, (-5e5, -5e5))

    # def OnView3DTriangles(self,event):
    #     if 'z' in self.pipeline.keys():
    #         if not 'glCanvas3D' in dir(self):
    #             #self.glCanvas3D = gl_render3D.LMGLCanvas(self)
    #             #self.AddPage(page=self.glCanvas3D, select=True, caption='3D')
    #             self.glCanvas3D = gl_render3D.showGLFrame()

    #         self.glCanvas3D.setTriang3D(self.pipeline['x'], 
    #                                   self.pipeline['y'], 
    #                                   self.pipeline['z'], 'z', 
    #                                   sizeCutoff=self.glCanvas3D.edgeThreshold)
                                      
    #         self.glCanvas3D.setCLim(self.glCanvas3D.clim, (0, 5e-5))

   
    def OnSaveMeasurements(self, event):
        fdialog = wx.FileDialog(None, 'Save measurements ...',
            wildcard='Numpy array|*.npy|Tab formatted text|*.txt', style=wx.SAVE)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath().encode()

            if outFilename.endswith('.txt'):
                of = open(outFilename, 'w')
                of.write('\t'.join(self.objectMeasures.dtype.names) + '\n')

                for obj in self.objectMeasures:
                    of.write('\t'.join([repr(v) for v in obj]) + '\n')
                of.close()

            else:
                np.save(outFilename, self.objectMeasures)



            
    @property
    def notebook(self):
        return self._mgr.GetNotebooks()[0]
            
    def _removeOldTabs(self):
        if not self.elv is None: #remove previous event viewer
            i = 0
            found = False
            while not found and i < self.notebook.GetPageCount():
                if self.notebook.GetPage(i) == self.elv:
                    self.notebook.DeletePage(i)
                    found = True
                else:
                    i += 1
                    
        if not self.colp is None: #remove previous colour viewer
            i = 0
            found = False
            while not found and i < self.notebook.GetPageCount():
                if self.notebook.GetPage(i) == self.colp:
                    self.notebook.DeletePage(i)
                    found = True
                else:
                    i += 1
                    
        if not self.mdp is None: #remove previous metadata viewer
            i = 0
            found = False
            while not found and i < self.notebook.GetPageCount():
                if self.notebook.GetPage(i) == self.mdp:
                    self.notebook.DeletePage(i)
                    found = True
                else:
                    i += 1
                    
    def _createNewTabs(self):
        #print 'md'
        self.mdp = MetadataTree.MetadataPanel(self, self.pipeline.mdh, editable=False)
        self.AddPage(self.mdp, caption='Metadata')
        
        #print 'cp'        
        if 'gFrac' in self.pipeline.filter.keys():
            self.colp = colourPanel.colourPanel(self, self.pipeline, self)
            self.AddPage(self.colp, caption='Colour', update=False)
            
        #print 'ev'
        if not self.pipeline.events is None:
            self.elv = eventLogViewer.eventLogPanel(self, self.pipeline.events, 
                                                        self.pipeline.mdh, 
                                                        [0, self.pipeline.selectedDataSource['tIndex'].max()])
    
            self.elv.SetCharts(self.pipeline.eventCharts)
            
            self.AddPage(self.elv, caption='Events', update=False)
            
        #print 'ud'
        self._mgr.Update()
            
        
            

            

    def OnOpenChannel(self, event):
        filename = wx.FileSelector("Choose a file to open", 
                                   nameUtils.genResultDirectoryPath(), 
                                   default_extension='h5r', 
                                   wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt')

        #print filename
        if not filename == '':
            self.OpenChannel(filename)

    def OnOpenRaw(self, event):
        from PYME.DSView import ViewIm3D, ImageStack
        ViewIm3D(ImageStack(), mode='visGUI', glCanvas=self.glCanvas)
        
    def AddExtrasMenuItem(self,label, callback):
        """Add an item to the VisGUI extras menu.
        
        parameters:
            label       textual label to use for the menu item.
            callback    function to call when user selects the menu item. This 
                        function should accept one argument, which will be the
                        wxPython event generated by the menu selection.
        """
        
        ID_NEWITEM = wx.NewId()
        self.extras_menu.Append(ID_NEWITEM, label)
        self.Bind(wx.EVT_MENU, callback, id=ID_NEWITEM)
        


   


    




class VisGuiApp(wx.App):
    def __init__(self, filename, use_shaders, *args):
        self.filename = filename
        self.use_shaders = use_shaders
        wx.App.__init__(self, *args)
        
        
    def OnInit(self):
        wx.InitAllImageHandlers()
        self.main = VisGUIFrame(None, self.filename, use_shaders=self.use_shaders)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True


def main_(filename=None, use_shaders=False):
    if filename == "":
        filename = None
    application = VisGuiApp(filename, use_shaders, 0)
    application.MainLoop()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="file that should be used")
    parser.add_argument('--use_shaders', dest="use_shaders", action='store_true', default=False,
                        help='switch shaders on(default: off)')
    parser.add_argument('--no_use_shaders', dest="use_shaders", action='store_false',
                        default=False, help='switch shaders off(default: off)')
    args = parser.parse_args()
    return args
    
def main():
    from multiprocessing import freeze_support
    freeze_support()
    
    filename = None
    args = parse()
    if wx.GetApp() is None: #check to see if there's already a wxApp instance (running from ipython -pylab or -wthread)
        main_(args.file, use_shaders=args.use_shaders)
    else:
        #time.sleep(1)
        visFr = VisGUIFrame(None, filename, False)
        visFr.Show()
        visFr.RefreshView()
        
if __name__ == '__main__':
    #from PYME.util import mProfile
    #mProfile.profileOn(['multiviewMapping.py', 'pyDeClump.py'])
    main()
    #mProfile.report()


