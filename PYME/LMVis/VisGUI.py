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
from PYME.misc import big_sur_fix
import os
os.environ['ETS_TOOLKIT'] = 'wx'

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
# import pylab

from PYME import config
from PYME.misc import extraCMaps
from PYME.IO.FileUtils import nameUtils

#import os
#from PYME.LMVis import gl_render3D

from PYME.LMVis import colourPanel
#from PYME.LMVis import renderers
from PYME.LMVis import pipeline

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR) #clobber unhelpful matplotlib debug messages
logging.getLogger('matplotlib.backends.backend_wx').setLevel(logging.ERROR)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

from PYME.ui import MetadataTree
from PYME.recipes import recipeGui
from PYME.recipes import modules #force modules (including 3rd party) to load

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
                 size=(900,750), style=wx.DEFAULT_FRAME_STYLE, use_shaders=True, cmd_args=None, pipeline_vars = {}):

        # populate about box info
        self._component_name = 'PYMEVisualise'
        self._long_desc = "Visualisation of localisation microscopy data."
        
        AUIFrame.__init__(self, parent, id, title, pos, size, style)
        
        
        self.cmd_args = cmd_args
        self._flags = 0
        
        self.pipeline = pipeline.Pipeline(visFr=self)
        self.pipeline.dataSources.update(pipeline_vars)

        visCore.VisGUICore.__init__(self, use_shaders=use_shaders)
        
        #self.Quads = None
               
        #self.SetMenuBar(self.CreateMenuBar())
        self.CreateMenuBar(use_shaders=use_shaders)

        self.statusbar = self.CreateStatusBar(1, wx.STB_SIZEGRIP)

        self.statusbar.SetStatusText("", 0)
       
        #self._leftWindow1 = wx.Panel(self, -1, size = wx.Size(220, 1000))
        #self._pnl = 0
        
        #initialize the common parts
        ###############################
        #NB: this has to come after the shell has been generated, but before the fold panel
        

        ################################   

        self.MainWindow = self #so we can access from shell
        self.sh = wx.py.shell.Shell(id=-1,
                                    parent=self, size=wx.Size(-1, -1), style=0, locals=self.__dict__,
                                    startupScript=config.get('VisGUI-console-startup-file', None),
              introText='PYMEVisualize - note that help, license, etc. below is for Python, not PYME\n\n')

        #self._mgr.AddPane(self.sh, aui.AuiPaneInfo().
        #                  Name("Shell").Caption("Console").Centre().CloseButton(False).CaptionVisible(False))

        self.AddPage(self.sh, caption='Shell')
             
        
        self.elv = None
        self.colp = None
        self.mdp = None
        self.rav = None

        self.generatedImages = []
        
        self.sh.Execute('from pylab import *')
        self.sh.Execute('from PYME.DSView.dsviewer import View3D')
        
        import os
        if os.getenv('PYMEGRAPHICSFIX'): # fix issue with graphics freezing on some machines (apparently matplotlib related)
            self.sh.Execute('plot()')
            self.sh.Execute('close()')

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
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        #self.Bind(wx.EVT_IDLE, self.OnIdle)
        #self.refv = False

        statusLog.SetStatusDispFcn(self.SetStatus)
        
        
        self.paneHooks.append(self.GenPanels)
        self.CreateFoldPanel()

        #from .layer_panel import CreateLayerPane, CreateLayerPanel
        #CreateLayerPane(sidePanel, self)
        #CreateLayerPanel(self)
        
        self._recipe_manager = recipeGui.PipelineRecipeManager(self.pipeline)
        self._recipe_editor = recipeGui.RecipeView(self, self._recipe_manager)
        self.AddPage(page=self._recipe_editor, select=False, caption='Pipeline Recipe')
        
        self.AddMenuItem('Recipe', 'Reconstruct from open image', self.reconstruct_pipeline_from_open_image)
        self.AddMenuItem('Recipe', 'Reconstruct from image file', self.reconstruct_pipeline_from_image_file)

        if not filename is None:
            def _recipe_callback():
                recipe = getattr(self.cmd_args, 'recipe', None)
                print('Using recipe: %s' % recipe)
                if recipe:
                    from PYME.recipes import modules
                    self.pipeline.recipe.update_from_yaml(recipe)
                    #self.recipeView.SetRecipe(self.pipeline.recipe)
                    self.update_datasource_panel()

                self._recipe_editor.update_recipe_text()
            
            wx.CallLater(50,self.OpenFile,filename, recipe_callback=_recipe_callback)
            #self.refv = False
        
        wx.CallAfter(self.RefreshView)

        nb = self._mgr.GetNotebooks()[0]
        nb.SetSelection(0)
        self.add_common_menu_items()
        
    def reconstruct_pipeline_from_image(self, image):
        self._recipe_manager.load_recipe_from_mdh(image.mdh)
        self.pipeline.selectDataSource(image.mdh['Pipeline.SelectedDataSource'])
        
    def reconstruct_pipeline_from_open_image(self, event=None):
        from PYME.IO import image
        names = image.openImages.keys()
    
        dlg = wx.SingleChoiceDialog(self.dsviewer, 'Select an image', 'Reconstruct pipeline from image', names)
    
        if dlg.ShowModal() == wx.ID_OK:
            #store a copy in the image for measurements etc ...
        
            im = image.openImages[names[dlg.GetSelection()]]
            
            self.reconstruct_pipeline_from_image(im)
            
    def reconstruct_pipeline_from_image_file(self, event=None, filename=None):
        from PYME.DSView import ImageStack
        im = ImageStack(filename=filename)

        self.reconstruct_pipeline_from_image(im)
        

    def OnMove(self, event):
        self.Refresh()
        self.Update()
        event.Skip()      

    def OnClose(self, event):
        while len(self.pipeline.filesToClose) > 0:
            self.pipeline.filesToClose.pop().close()

        # pylab.close('all')
        matplotlib.pyplot.close('all')
        self._cleanup()
        
        #AUIFrame.OnQuit(self, event)

    def OnDocumentation(self, event):
        import webbrowser
        webbrowser.open('https://python-microscopy.org/doc/')
        event.Skip()

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
            wildcard='Numpy array|*.npy|Tab formatted text|*.txt', style=wx.FD_SAVE)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath()

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
            
    # def _removeOldTabs(self):
    #     if not self.elv is None: #remove previous event viewer
    #         i = 0
    #         found = False
    #         while not found and i < self.notebook.GetPageCount():
    #             if self.notebook.GetPage(i) == self.elv:
    #                 self.notebook.DeletePage(i)
    #                 found = True
    #             else:
    #                 i += 1
    #
    #     if not self.colp is None: #remove previous colour viewer
    #         i = 0
    #         found = False
    #         while not found and i < self.notebook.GetPageCount():
    #             if self.notebook.GetPage(i) == self.colp:
    #                 self.notebook.DeletePage(i)
    #                 found = True
    #             else:
    #                 i += 1
    #
    #     if not self.mdp is None: #remove previous metadata viewer
    #         i = 0
    #         found = False
    #         while not found and i < self.notebook.GetPageCount():
    #             if self.notebook.GetPage(i) == self.mdp:
    #                 self.notebook.DeletePage(i)
    #                 found = True
    #             else:
    #                 i += 1

    def _removeOldTabs(self):
        self.DeletePage(self.elv)
        self.elv = None
    
        self.DeletePage(self.colp)
        self.colp = None
        
        self.DeletePage(self.mdp)
        self.mdp = None
        
                    
    def _createNewTabs(self):
        logger.debug('Creating tabs')
        self.adding_panes = True
        self.mdp = MetadataTree.MetadataPanel(self, self.pipeline.mdh, editable=False)
        self.AddPage(self.mdp, caption='Metadata', select=False, update=False)
        
        #print 'cp'        
        if 'gFrac' in self.pipeline.filter.keys():
            self.colp = colourPanel.colourPanel(self, self.pipeline, self)
            self.AddPage(self.colp, caption='Colour', select=False, update=False)
            
        #print 'ev'
        if not self.pipeline.events is None:
            #self.elv = eventLogViewer.eventLogPanel(self, self.pipeline.events,
            #                                            self.pipeline.mdh,
            #                                            [0, self.pipeline.selectedDataSource['tIndex'].max()])

            st = min(self.pipeline.events['Time'].min() - self.pipeline.mdh['StartTime'], 0)
            et = 1.1*self.pipeline.selectedDataSource['tIndex'].max()*self.pipeline.mdh['Camera.CycleTime']
            print(st, et)
            
            self.elv = eventLogViewer.eventLogTPanel(self, self.pipeline.events,self.pipeline.mdh,[st, et])
    
            self.elv.SetCharts(self.pipeline.eventCharts)
            
            self.AddPage(self.elv, caption='Events', select=False, update=False)
            self.elv.activate() #turn painting on now to avoid a paint when we create

        logger.debug('Finished creating tabs')
        self.adding_panes = False
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
        from PYME.IO import image
        from PYME.DSView import ViewIm3D
        try:
            ViewIm3D(image.ImageStack(haveGUI=True), mode='visGUI', glCanvas=self.glCanvas)
        except image.FileSelectionError:
            # the user canceled the open dialog
            pass
        
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
    def __init__(self, filename, use_shaders, cmd_args, *args):
        self.filename = filename
        self.use_shaders = use_shaders
        self.cmd_args = cmd_args
        wx.App.__init__(self, *args)
        
        
    def OnInit(self):
        self.main = VisGUIFrame(None, self.filename, use_shaders=self.use_shaders, cmd_args=self.cmd_args)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True


def main_(filename=None, use_shaders=False, args=None):
    from PYME.misc import check_for_updates
    if filename == "":
        filename = None
    application = VisGuiApp(filename, use_shaders, args, 0)
    check_for_updates.gui_prompt_once()
    application.MainLoop()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="file that should be used", default=None, nargs='?')
    parser.add_argument('-r', '--recipe', help='recipe to use for variable portion of pipeline', dest='recipe', default=None)
    parser.add_argument('-s', '--use-shaders', dest="use_shaders", action='store_true', default=True,
                        help='switch shaders on(default: off)')
    parser.add_argument('--no-shaders', dest="use_shaders", action='store_false',
                        default=True, help='switch shaders off(default: off)')
    parser.add_argument('--new-layers', dest='new_layers', action='store_true', default=True)
    parser.add_argument('--no-layers', dest='new_layers', action='store_false', default=True)
    args = parser.parse_args()
    return args
    
def main():
    from multiprocessing import freeze_support
    import PYME.config
    freeze_support()
    
    filename = None
    args = parse()
    
    PYME.config.config['VisGUI-new_layers'] = args.new_layers
    
    if wx.GetApp() is None: #check to see if there's already a wxApp instance (running from ipython -pylab or -wthread)
        main_(args.file, use_shaders=args.use_shaders, args=args)
    else:
        #time.sleep(1)
        visFr = VisGUIFrame(None, args.file, args.use_shaders)
        visFr.Show()
        visFr.RefreshView()
        
if __name__ == '__main__':
    from PYME.util import mProfile
    mProfile.profileOn(['pipeline.py', 'tabular.py'])
    main()
    mProfile.report()

def ipython_visgui(filename=None, **kwargs):
    import PYME.config
    
    if wx.GetApp() is None:
        raise RuntimeError('No wx App instance found. Start one using the `\%gui wx` magic in ipython before running this command')

    PYME.config.config['VisGUI-new_layers'] = True
    
    visFr = VisGUIFrame(None, filename=filename, pipeline_vars = kwargs)
    visFr.Show()
    return visFr
    
def ipython_pymevisualize(filename=None, **kwargs):
    return ipython_visgui(filename, **kwargs)
