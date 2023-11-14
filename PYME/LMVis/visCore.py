# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:54:52 2016

@author: david
"""
import wx
import wx.py.shell

#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import wx.lib.agw.aui as aui

#hacked so py2exe works
#from PYME.DSView.dsviewer import View3D

#from PYME.LMVis import workspaceTree
from PYME.IO.FileUtils import nameUtils

import os

#from PYME.LMVis import colourPanel
from PYME.LMVis import renderers

import logging
logger = logging.getLogger(__name__)

import PYME.config

#try importing our drift correction stuff
# disabled as drift correction now available in a recipe friendly way
HAVE_DRIFT_CORRECTION = False
#try:
#    from PYMEnf.DriftCorrection.driftGUI import CreateDriftPane
#    HAVE_DRIFT_CORRECTION = True
#    #from PYMEnf.DriftCorrection import driftGUI
#    #renderers.renderMetadataProviders.append(driftGUI.dp.SaveMetadata)
#except:
#    pass

from PYME.LMVis.colourFilterGUI import CreateColourFilterPane
from PYME.LMVis import displayPane
from PYME.ui.filterPane import CreateFilterPane

from PYME.LMVis import pointSettingsPanel
from PYME.LMVis import quadTreeSettings
from PYME.LMVis import triBlobs

#from PYME.Analysis import MetadataTree
from PYME.contrib import dispatch
import numpy as np
#import scipy.special

#from PYME.DSView import eventLogViewer



from PYME.LMVis import statusLog
#from PYME.recipes import recipeGui

class VisGUICore(object):
    @property
    def _win(self):
        """
        Returns the window oject associated with this VisGUICore instance.

        This can either be a window instance (in PYMEVis) or a non-window "manager" in the LMDisplay PYMEImage plugin. We sometimes
        need to access the window to create controls etc ...
        """
        if isinstance(self, wx.Window):
            return self
        else:
            return self.dsviewer
        
    def __init__(self, use_shaders=False):
        self._new_layers = PYME.config.get('VisGUI-new_layers', True)
        self.viewMode = 'points' #one of points, triangles, quads, or voronoi
        #self.colData = 't'
        self.pointDisplaySettings = pointSettingsPanel.PointDisplaySettings()
        self.pointDisplaySettings.on_trait_change(self.RefreshView)
        
        self.quadTreeSettings = quadTreeSettings.QuadTreeSettings()
        self.quadTreeSettings.on_trait_change(self.RefreshView)
        
        self.pipeline.blobSettings.on_trait_change(self.RefreshView)
        self.pipeline.onRebuild.connect(self.RefreshView)
        
        #initialize the gl canvas
        

        gl_pan = wx.Panel(self._win)
        sizer = wx.BoxSizer(wx.VERTICAL)


        
        if not use_shaders:
            from PYME.LMVis import gl_render3D
            self.glCanvas = gl_render3D.LMGLCanvas(gl_pan)
        else:
            from PYME.LMVis.gl_render3D_shaders import LMGLShaderCanvas, LegacyGLCanvas
            if self._new_layers:
                #use stripped down version
                self.glCanvas = LMGLShaderCanvas(gl_pan)
            else:
                self.glCanvas = LegacyGLCanvas(gl_pan)

        sizer.Add(self.create_tool_bar(gl_pan), 0, wx.EXPAND, 0)
        sizer.Add(self.glCanvas, 5, wx.EXPAND, 0)
        gl_pan.SetSizerAndFit(sizer)
        self._win.AddPage(page=gl_pan, caption='View')#, select=True)

        #self.glCanvas.setCMap(pylab.cm.gist_rainbow) #pylab.cm.hot

        #self.rec_gui = recipeGui.
        #win.AddPage(page=self.glCanvas, caption='View')#, select=True)
        
        self.refv = False
        
        self._legacy_layer = None
        
        self.layer_added = dispatch.Signal()
        
        renderers.renderMetadataProviders.append(self.SaveMetadata)
        self.use_shaders = use_shaders
        
        wx.CallLater(100, self.OnIdle)
        
    
    def OnIdle(self, event=None):
        """ Refresh the glDisplay *after* all windows have been created and data loaded.
       
        TODO - rename, as the OnIdle name is a historical artifact (was originally called using an wx.EVT_IDLE binding, 
        now uses wx.CallLater with a delay.
        """
        #logger.debug('Ev Idle')
        if self.glCanvas._is_initialized and not self.refv:
            self.refv = True
            #logger.debug((self.viewMode, self.pointDisplaySettings.colourDataKey))
            self.SetFit()
           
            if self._new_layers:
                pass
               # if self.pipeline.ready and not len(self.layers) > 0:
               #     l = self.add_layer(method='points')
               #     if 't' in self.pipeline.keys():
               #         l.engine.set(vertexColour='t')
               #     elif 'z' in self.pipeline.keys():
               #         l.engine.set(vertexColour='t')
            else:
                self.RefreshView()
                self.displayPane.OnPercentileCLim(None)
                
            self.glCanvas.Refresh()
            self.glCanvas.Update()
            logger.debug('Refreshed glCanvas after load')
            
    def GenPanels(self, sidePanel):
        logger.debug('GenPanels')
        self.GenDataSourcePanel(sidePanel)
        
        #if HAVE_DRIFT_CORRECTION:
        #    self.driftPane = CreateDriftPane(sidePanel, self.pipeline.mapping, self.pipeline)

        self.filterPane = CreateFilterPane(sidePanel, self.pipeline.filterKeys, self.pipeline, self)

        if self._new_layers:
            #self.colourFilterPane = CreateColourFilterPane(sidePanel, self.pipeline.colourFilter, self.pipeline)
            #self.displayPane = displayPane.CreateDisplayPane(sidePanel, self.glCanvas, self)
            #self.displayPane.Bind(displayPane.EVT_DISPLAY_CHANGE, self.RefreshView)
        
        
            from .layer_panel import CreateLayerPane, CreateLayerPanel
            self._layer_pane = CreateLayerPane(sidePanel, self)
            #CreateLayerPanel(self)
            
            if self.use_shaders:
                from .view_clipping_pane import GenViewClippingPanel
                GenViewClippingPanel(self, sidePanel)
        else:
            self.colourFilterPane = CreateColourFilterPane(sidePanel, self.pipeline.colourFilter, self.pipeline)
            self.displayPane = displayPane.CreateDisplayPane(sidePanel, self.glCanvas, self)
            self.displayPane.Bind(displayPane.EVT_DISPLAY_CHANGE, self.RefreshView)
            
            if self.viewMode == 'quads':
                quadTreeSettings.GenQuadTreePanel(self, sidePanel)
    
            if self.viewMode in ['points', 'tracks', 'pointsprites', 'shadedpoints']:
                pointSettingsPanel.GenPointsPanel(self, sidePanel)
            if self.viewMode == 'blobs':
                triBlobs.GenBlobPanel(self, sidePanel)
    
            if self.viewMode == 'interp_triangles':
                pointSettingsPanel.GenPointsPanel(self, sidePanel,'Vertex Colours')

        
        self.glCanvas.Refresh()
        
    def GenDataSourcePanel(self, pnl):
        from PYME.recipes.vertical_recipe_display import RecipeDisplayPanel
        
        logger.debug('Creating datasource panel')
        item = afp.foldingPane(pnl, -1, caption="Data Pipeline", pinned = True)

        pan = wx.Panel(item, -1)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'output:'), 0, wx.ALL, 2)
        self.chSource = wx.Choice(pan, -1, choices=[])
        hsizer.Add(self.chSource, 1, wx.ALL | wx.EXPAND, 2)
        pan.SetSizerAndFit(hsizer)
        self.update_datasource_panel()
        self.chSource.Bind(wx.EVT_CHOICE, self.OnSourceChange)
        self.pipeline.onRebuild.connect(self.update_datasource_panel)

        item.AddNewElement(pan, foldable=False)

        self.recipeView = RecipeDisplayPanel(item)
        self.recipeView.SetRecipe(self.pipeline.recipe)
        item.AddNewElement(self.recipeView, priority=20, foldable=False)

        pnl.AddPane(item, 20)
        
    def update_datasource_panel(self, event=None, **kwargs):
        dss = list(self.pipeline.dataSources.keys())
        self.chSource.SetItems(dss)
        if not self.pipeline.selectedDataSourceKey is None:
            self.chSource.SetStringSelection(self.pipeline.selectedDataSourceKey)

        try:
            self.Layout()
        except AttributeError:
            logger.debug('No Layout method') 
            pass
        

    def OnSourceChange(self, event):
        self.pipeline.selectDataSource(self.chSource.GetStringSelection())
        
        
    def pointColour(self):
        pointColour = None
        
        colData = self.pointDisplaySettings.colourDataKey
        
        if colData == '<None>':
            pointColour = None
        elif not self.pipeline.colourFilter is None:
            if colData in self.pipeline.keys():
                pointColour = self.pipeline[colData]
            elif colData in self.pipeline.GeneratedMeasures.keys():
                pointColour = self.pipeline.GeneratedMeasures[colData]
            else:
                pointColour = None

        return pointColour
        
    def CreateMenuBar(self, subMenu = False, use_shaders = False):
        logger.debug('Creating VisGUI menu bar')
        if 'dsviewer' in dir(self):
            parent = self.dsviewer
        else:
            parent = self

        self.AddMenuItem('File', '&Open', self.OnOpenFile)
        if not subMenu:
            self.AddMenuItem('File', "Open &Raw/Prebleach Data", self.OnOpenRaw)
            self.AddMenuItem('File', "Open Extra &Channel", self.OnOpenChannel)
            
        self.AddMenuItem('File', 'Save filtered localizations', self.OnSave)
        
        if not subMenu:
            self.AddMenuItem('File', itemType='separator')
            self.AddMenuItem('File', "&Save Measurements", self.OnSaveMeasurements)

            #self.AddMenuItem('File', itemType='separator')
            #self.AddMenuItem('File', "&Exit", self.OnQuit,id = wx.ID_EXIT)


        if not self._new_layers:
            self.AddMenuItem('View', '&Points', self.OnViewPoints, itemType='normal') #TODO - add radio type
            if use_shaders:
                self.AddMenuItem('View', '&Pointsprites', self.OnViewPointsprites)
                self.AddMenuItem('View', '&Shaded Points', self.OnViewShadedPoints)
            
            self.AddMenuItem('View',  '&Triangles', self.OnViewTriangles)
            self.AddMenuItem('View', '3D Triangles', self.OnViewTriangles3D)
            self.AddMenuItem('View', '&Quad Tree', self.OnViewQuads)
            if not use_shaders:
                self.AddMenuItem('View', '&Voronoi', self.OnViewVoronoi)
                self.AddMenuItem('View', '&Interpolated Triangles', self.OnViewInterpTriangles)
                self.AddMenuItem('View', '&Blobs', self.OnViewBlobs)
                self.AddMenuItem('View', '&Tracks', self.OnViewTracks)
    
    
            #self.view_menu.Check(ID_VIEW_POINTS, True)
            #self.view_menu.Enable(ID_VIEW_QUADS, False)
    
            self.AddMenuItem('View', itemType='separator')
        
        self.AddMenuItem('View', '&Home\tAlt-H', self.OnHome)
        self.AddMenuItem('View', '&Fit\tAlt-F', self.SetFit)
        self.AddMenuItem('View', 'Fit ROI\tAlt-8', self.OnFitROI)

        #this needs an ID as we bind to it elsewhere (in the filter panel)
        self.ID_VIEW_CLIP_ROI = wx.NewIdRef()
        self.AddMenuItem('View', 'Clip to ROI\tF8', id=self.ID_VIEW_CLIP_ROI)

        self.AddMenuItem('View', 'Re&center\tAlt-C', self.OnRecenter)
        self.AddMenuItem('View', 'Reset &rotation\tAlt-R', self.OnResetRotation)

        self.AddMenuItem('View', itemType='separator')
        
        renderers.init_renderers(self)

        from PYME.LMVis import Extras
        Extras.InitPlugins(self)
        
        try:
            #see if we can find any 'non free' plugins
            from PYMEnf.Analysis.LMVis import Extras
            Extras.InitPlugins(self)
        except ImportError:
            pass

        if not subMenu:
            self.AddMenuItem('Help', "&Documentation", self.OnDocumentation)
            #self.AddMenuItem('Help', "&About", self.OnAbout)
            
    def create_tool_bar(self, parent):
        from .displayPane import DisplayPaneHorizontal
        
        return DisplayPaneHorizontal(parent, self.glCanvas, self)
        
        
    def OnViewPoints(self,event):
        self.viewMode = 'points'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewPointsprites(self, event):
        self.viewMode = 'pointsprites'
        # self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewShadedPoints(self,event):
        NO_NORMALS_MSG = '''Shaded points is experimental and only works for datasets with point normals (xn, yn, zn) defined.
        The assignment of normals requires a concept of a surface to which the point belongs, and is not trivial for most
        localization datasets - i.e. this rendering mode is usually not applicable. Data with normals will typically be
        the output of surface estimation routines, or derived from some external meshed dataset (e.g. the OpenGL teapot).'''
        
        if not 'xn' in self.pipeline.keys():
            wx.MessageBox(NO_NORMALS_MSG, 'Cannot use shaded points for this data', style=wx.OK)
            return
        
        self.viewMode = 'shadedpoints'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewTracks(self,event):
        self.viewMode = 'tracks'
        #self.glCanvas.cmap = pylab.cm.hsv
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewBlobs(self,event):
        self.viewMode = 'blobs'
        self.RefreshView()
        #self.CreateFoldPanel()
        #self.OnPercentileCLim(None)

    def OnViewTriangles(self,event):
        self.viewMode = 'triangles'
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewTriangles3D(self,event):
        self.viewMode = 'triangles3D'
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewQuads(self,event):
        self.viewMode = 'quads'
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewVoronoi(self,event):
        self.viewMode = 'voronoi'
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)

    def OnViewInterpTriangles(self,event):
        self.viewMode = 'interp_triangles'
        self.RefreshView()
        #self.CreateFoldPanel()
        self.displayPane.OnPercentileCLim(None)
        
    def OnOpenFile(self, event):
        filename = wx.FileSelector("Choose a file to open", 
                                   nameUtils.genResultDirectoryPath(), 
                                   wildcard='All supported formats|*.h5r;*.txt;*.mat;*.csv;*.hdf;*.3d;*.3dlp|PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt|Matlab data (*.mat)|*.mat|Comma separated values (*.csv)|*.csv|HDF Tabular (*.hdf)|*.hdf')

        #print filename
        if not filename == '':
            self.OpenFile(filename)
            
    def OnSave(self, event):
        filename = wx.SaveFileSelector("Save pipeline output as ...", '.hdf')
        if not filename == '':
            self.pipeline.save_hdf(filename)
            
    def RegenFilter(self):
        logger.warn('RegenFilter is deprecated, please use pipeline.Rebuild() instead.')
        self.pipeline.Rebuild()
        
    def add_pointcloud_layer(self, method='points', ds_name='output', **kwargs):
        #from .layer_wrapper import LayerWrapper
        from .layers.pointcloud import PointCloudRenderLayer
        l = PointCloudRenderLayer(self.pipeline, method=method, dsname=ds_name, **kwargs)
        self.add_layer(l)

        logger.debug('Added layer, datasouce=%s' % l.dsname)
        return l

    def add_mesh_layer(self, method='shaded', ds_name=None, **kwargs):
        from PYME.LMVis.layers.mesh import TriangleRenderLayer
        from PYME.misc.colormaps import cm
        if ds_name is None:
            from PYME.experimental._triangle_mesh import TriangleMesh
            mesh_ds = [k for k, d in self.pipeline.dataSources.items() if isinstance(d, TriangleMesh)]
            ds_name = mesh_ds[-1]
        ds_stub = ds_name.rstrip('0123456789')
        _, surf_count = self.pipeline.new_ds_name(ds_stub, return_count=True)
        surf_count -= 1  # To match current count
        l = TriangleRenderLayer(self.pipeline, dsname=ds_name, method=method, cmap = cm.solid_cmaps[surf_count % len(cm.solid_cmaps)])
        self.add_layer(l)

        logger.debug('Added layer, datasouce=%s' % l.dsname)
        return l

    def add_quiver_layer(self, ds_name=None, **kwargs):
        from PYME.LMVis.layers.quiver import QuiverRenderLayer
        from PYME.misc.colormaps import cm
        if ds_name is None:
            from PYME.experimental._triangle_mesh import TriangleMesh
            mesh_ds = [k for k, d in self.pipeline.dataSources.items() if isinstance(d, TriangleMesh)]
            ds_name = mesh_ds[-1]
        
        l = QuiverRenderLayer(self.pipeline, dsname=ds_name)
        self.add_layer(l)

        logger.debug('Added layer, datasouce=%s' % l.dsname)
        return l

    def add_layer(self, layer):
        self.glCanvas.layers.append(layer)
        self.glCanvas.recenter_bbox()
        layer.on_update.connect(self.glCanvas.refresh)
        self.glCanvas.refresh()
        
        wx.CallAfter(self.layer_added.send, self)

    @property
    def layers(self):
        return self.glCanvas.layers
        
    def RefreshView(self, event=None, **kwargs):
        #self.CreateFoldPanel()
        if not self.pipeline.ready:
            return #get out of here
        
        if self._new_layers:
            #refresh view no longer updates the display
            
            #FIXME - this doesn't belong here (used to be done in SetPoints)
            self.glCanvas.view.translation[2] = self.pipeline['z'].mean()
            return
        
        
        #FIXME - this should be handled within the filter pane
        self.filterPane.stFilterNumPoints.SetLabel('%d of %d events' % (len(self.pipeline.filter['x']), len(self.pipeline.selectedDataSource['x'])))

        if len(self.pipeline['x']) == 0:
            self.glCanvas.setOverlayMessage('No data points - try adjusting the filter')
            return
        else:
            self.glCanvas.setOverlayMessage('')

        if not self.glCanvas._is_initialized: #glcanvas is not initialised
            return

        #bCurr = wx.BusyCursor()

        #delete previous layers (new view)
        # self.glCanvas.layers = []
        
        # only delete the layer we created on the last call - leave other layers alone.
        if self._legacy_layer:
            try:
                self.glCanvas.layers.remove(self._legacy_layer)
            except ValueError:
                pass
            self._legacy_layer = None
        
        self.glCanvas.pointSize = self.pointDisplaySettings.pointSize

        if self.pipeline.objects is None:
#            if 'bObjMeasure' in dir(self):
#                self.bObjMeasure.Enable(False)
            self.objectMeasures = None

            if 'rav' in dir(self) and not self.rav is None: #remove previous event viewer
                i = 0
                found = False
                while not found and i < self.notebook.GetPageCount():
                    if self.notebook.GetPage(i) == self.rav:
                        self.notebook.DeletePage(i)
                        found = True
                    else:
                        i += 1
                        
                self.rav = None

        if self.viewMode == 'points':
            if 'setPoints3D' in dir(self.glCanvas) and 'z' in self.pipeline.keys():
                #new mode
                self.glCanvas.setPoints3D(self.pipeline['x'], 
                                      self.pipeline['y'], 
                                      self.pipeline['z'], 
                                      self.pointColour(), alpha=self.pointDisplaySettings.alpha)
            else:
                self.glCanvas.setPoints(self.pipeline['x'], 
                                    self.pipeline['y'], self.pointColour())
        elif self.viewMode == 'pointsprites':
            self.glCanvas.setPoints3D(self.pipeline['x'],
                                      self.pipeline['y'],
                                      self.pipeline['z'],
                                      self.pointColour(), alpha=self.pointDisplaySettings.alpha, mode='pointsprites')
        elif self.viewMode == 'shadedpoints':
            try:
                self.glCanvas.setPoints3D(self.pipeline['x'],
                                          self.pipeline['y'],
                                          self.pipeline['z'],
                                          self.pointColour(),
                                          alpha=self.pointDisplaySettings.alpha,
                                          normal_x=self.pipeline['normal_x'],
                                          normal_y=self.pipeline['normal_y'],
                                          normal_z=self.pipeline['normal_z'],
                                          mode='shadedpoints')
            except KeyError:
                self.glCanvas.setPoints3D(self.pipeline['x'],
                                          self.pipeline['y'],
                                          self.pipeline['z'],
                                          self.pointColour(),
                                          alpha=self.pointDisplaySettings.alpha,
                                          normal_x=self.pipeline['xn'],
                                          normal_y=self.pipeline['yn'],
                                          normal_z=self.pipeline['zn'],
                                          mode='shadedpoints')
        elif self.viewMode == 'tracks':
            if 'setTracks3D' in dir(self.glCanvas) and 'z' in self.pipeline.keys():
                self.glCanvas.setTracks3D(self.pipeline['x'], 
                                    self.pipeline['y'], 
                                    self.pipeline['z'],
                                    self.pipeline['clumpIndex'], 
                                    self.pointColour())
            else:
                self.glCanvas.setTracks(self.pipeline['x'], 
                                    self.pipeline['y'], 
                                    self.pipeline['clumpIndex'], 
                                    self.pointColour())
                                    
        elif self.viewMode == 'triangles':
            self.glCanvas.setTriang(self.pipeline.getTriangles())

        elif self.viewMode == 'triangles3D':
            self.glCanvas.setTriang3D(self.pipeline['x'], 
                                      self.pipeline['y'], 
                                      self.pipeline['z'], 'z', 
                                      sizeCutoff=self.glCanvas.edgeThreshold)

        elif self.viewMode == 'voronoi':
            status = statusLog.StatusLogger("Generating Voronoi Diagram ... ")
            self.glCanvas.setVoronoi(self.pipeline.getTriangles())
            

        elif self.viewMode == 'quads':
            if self.pipeline.Quads is None:
                status = statusLog.StatusLogger("Generating QuadTree ...")
                self.pipeline.GenQuads()
                

            self.glCanvas.setQuads(self.pipeline.Quads)

        elif self.viewMode == 'interp_triangles':
            self.glCanvas.setIntTriang(self.pipeline.getTriangles(), self.pointColour())

        elif self.viewMode == 'blobs':
            if self.pipeline.objects is None:
                #check to see that we don't have too many points
                if len(self.pipeline['x']) > 1e5:
                    goAhead = wx.MessageBox('You have %d events in the selected ROI;\nThis could take a LONG time ...' % len(self.pipeline['x']), 'Continue with blob detection', wx.YES_NO|wx.ICON_EXCLAMATION)
    
                    if not goAhead == wx.YES:
                        return

            self.glCanvas.setBlobs(*self.pipeline.getBlobs())
            self.objCInd = self.glCanvas.c
            
        #save the new layer we created so we can remove it
        self._legacy_layer = self.glCanvas.layers[-1]

        self.displayPane.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], 
                                        self.glCanvas.clim[1])

        if 'colp' in dir(self) and not self.colp is None and self.colp.IsShown():
            self.colp.refresh()

        #self.sh.shell.user_ns.update(self.__dict__)
        #wx.EndBusyCursor()
        #self.workspaceView.RefreshItems()
        
        
    def SetFit(self,event = None):
        xsc = self.pipeline.imageBounds.width()*1./self.glCanvas.Size[0]
        ysc = self.pipeline.imageBounds.height()*1./self.glCanvas.Size[1]

        if xsc != 0 and ysc != 0:
            if xsc > ysc:
                self.glCanvas.setView(self.pipeline.imageBounds.x0, self.pipeline.imageBounds.x1,
                                      self.pipeline.imageBounds.y0, self.pipeline.imageBounds.y0 + xsc*self.glCanvas.Size[1])
            else:
                self.glCanvas.setView(self.pipeline.imageBounds.x0, self.pipeline.imageBounds.x0 + ysc*self.glCanvas.Size[0],
                                      self.pipeline.imageBounds.y0, self.pipeline.imageBounds.y1)

    def OnFitROI(self,event = None):
        if 'x' in self.pipeline.filterKeys.keys():
            xbounds = self.pipeline.filterKeys['x']
        else:
            xbounds = (self.pipeline.imageBounds.x0, self.pipeline.imageBounds.x1)

        if 'y' in self.pipeline.filterKeys.keys():
            ybounds = self.pipeline.filterKeys['y']
        else:
            ybounds = (self.pipeline.imageBounds.y0, self.pipeline.imageBounds.y1)
        
        xsc = (xbounds[1] - xbounds[0])*1./self.glCanvas.Size[0]
        ysc = (ybounds[1] - ybounds[0])*1./self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(xbounds[0], xbounds[1], ybounds[0], 
                                  ybounds[0] + xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(xbounds[0], xbounds[0] + ysc*self.glCanvas.Size[0], 
                                  ybounds[0], ybounds[1])

    def OnRecenter(self, event=None):
        self.glCanvas.recenter_bbox()
        self.glCanvas.Refresh()

    def OnResetRotation(self, event=None):
        self.glCanvas.ResetView()
        self.glCanvas.Refresh()

    def OnHome(self, event=None):
        self.OnResetRotation(event)
        self.SetFit(event)

    def SetStatus(self, statusText):
        self.statusbar.SetStatusText(statusText, 0)
        
    def SaveMetadata(self, mdh):
        mdh['Filter.Keys'] = self.pipeline.filterKeys
        
        recipe = getattr(self.pipeline, 'recipe', None)
        
        if not recipe is None:
            mdh['Pipeline.Recipe'] = recipe.toYAML()
            mdh['Pipeline.SelectedDataSource'] = self.pipeline.selectedDataSourceKey
            
        
        #if HAVE_DRIFT_CORRECTION and 'x' in self.pipeline.mapping.mappings.keys(): #drift correction has been applied
        #    self.driftPane.dp.SaveMetadata(mdh)

    def AddMenuItem(self, menuName, *args, **kwargs):
        """ Add a menu item. Calls AUIFrame.AddMenuItem. Should be over-ridden when called from VisGUI, and only
        exposed / used when called from within a dsviewer module."""
        logger.debug('Calling AddMenuItem from visCore')
        self.dsviewer.AddMenuItem('Points>' + menuName, *args, **kwargs)
        
    def _create_base_layer(self):
        from PYME.misc.colormaps import cm
        from PYME.LMVis.layers import layer_defaults
        if self._new_layers and len(self.layers) == 0:
            #add a new layer
            l = self.add_pointcloud_layer(**layer_defaults.new_layer_settings('points', ds_keys=list(self.pipeline.keys())))
            # if 't' in self.pipeline.keys():
            #     l.set(vertexColour='t')
            # elif 'z' in self.pipeline.keys():
            #     l.set(vertexColour='z')
                
        colour_chans = self.pipeline.colourFilter.getColourChans()
        if len(colour_chans) > 1:
            #add a layer for each colour channel
            for i, c in enumerate(sorted(colour_chans)):
                #self.add_pointcloud_layer(ds_name=('output.' + c), cmap=cm.solid_cmaps[i % len(cm.solid_cmaps)], visible=False)
                self.add_pointcloud_layer(ds_name=('output.' + c), **layer_defaults.new_layer_settings('points_channel', i, overrides=dict(visible=False)))
                
    def _populate_open_args(self, filename):
        from PYME.warnings import warn
        args = {}
    
        if os.path.splitext(filename)[1] == '.h5r':
            pass
        elif os.path.splitext(filename)[1] == '.hdf':
            pass
        elif os.path.splitext(filename)[1] == '.mat':
            from PYME.LMVis import importTextDialog
            from scipy.io import loadmat
        
            mf = loadmat(filename)
            if ('x' not in mf.keys()) or ('y' not in mf.keys()):
                # This MATLAB file has some weird variable names

                if (len([k for k in mf.keys() if not k.startswith('_')]) < 3):
                    # All the data is probably packed in a single variable
                    dlg = importTextDialog.ImportMatDialog(self, [k for k in mf.keys() if not k.startswith('__')])
                    ret = dlg.ShowModal()
                
                    if not ret == wx.ID_OK:
                        dlg.Destroy()
                        logger.info("opening Matlab file was canceled")
                        return #we cancelled
                
                    args['FieldNames'] = dlg.GetFieldNames()
                    args['VarName'] = dlg.GetVarName()
                    # args['PixelSize'] = dlg.GetPixelSize()
            
            
                    dlg.Destroy()
                else:
                    # We have to map the field names
                    from PYME.LMVis import importTextDialog

                    dlg = importTextDialog.ImportMatlabDialog(self, mf)

                    ret = dlg.ShowModal()
                
                    if not ret == wx.ID_OK:
                        dlg.Destroy()
                        logger.info("opening Matlab file was canceled")
                        return #we cancelled

                    args['FieldNames'] = dlg.GetFieldNames()
                    args['PixelSize'] = dlg.GetPixelSize()
                    args['Multichannel'] = dlg.GetMultichannel()
                
                    dlg.Destroy()

        else: #assume it's a text file
            from PYME.LMVis import importTextDialog
            from PYME.IO import csv_flavours
        
            dlg = importTextDialog.ImportTextDialog(self, filename)
            ret = dlg.ShowModal()
        
            if not ret == wx.ID_OK:
                dlg.Destroy()
                logger.info("opening Text/CSV file was canceled")
                warn('Open file was canceled by user') # example how we could bring up a message box
                return #we cancelled
            
            text_options = {'columnnames': dlg.GetFieldNames(),
                            'skiprows' : dlg.GetNumberComments(),
                            'delimiter' : dlg.GetDelim(),
                            'invalid_raise' : not csv_flavours.csv_flavours[dlg.GetFlavour()].get('ignore_errors', False),
                            }
        
            #args['FieldNames'] = dlg.GetFieldNames()
            # remove trailing whitespace/line brake on last field name
            #args['FieldNames'][-1] = args['FieldNames'][-1].rstrip()
            #args['SkipRows'] = dlg.GetNumberComments()
            args['text_options'] = text_options
            args['PixelSize'] = dlg.GetPixelSize()
        
            #print 'Skipping %d rows' %args['SkipRows']
            dlg.Destroy()
            
        return args

    def OpenFile(self, filename, recipe_callback=None):
        # get rid of any old layers
        while len(self.layers) > 0:
            self.layers.pop()
        
        logger.debug('Creating Pipeline')
        if filename is None and not ds is None:
            self.pipeline.OpenFile(ds=ds)
        else:
            args = self._populate_open_args(filename)
            if args is None:
                return
            self.pipeline.OpenFile(filename, **args)
        logger.debug('Pipeline Created')
        
        #############################
        #now do all the gui stuff
        self.recipeView.invalidate_layout()
        self.update_datasource_panel()
        
        if isinstance(self, wx.Frame):
            #run this if only we are the main frame
            self.SetTitle('PYME Visualise - ' + filename)
            self._removeOldTabs()
            self._createNewTabs()
            
            #self.CreateFoldPanel()
            logger.debug('Gui stuff done')
        
        try:
            if recipe_callback:
                recipe_callback()
            
        finally:
            self.SetFit()
        
        
            wx.CallLater(100, self._create_base_layer)
            #wx.CallAfter(self.RefreshView)

    def OpenChannel(self, filename, recipe_callback=None, channel_name=''):
        args = self._populate_open_args(filename)
    
        logger.debug('Creating Pipeline')
        self.pipeline.OpenChannel(filename, channel_name=channel_name, **args)
        logger.debug('Pipeline Created')
    
        #############################
        #now do all the gui stuff
    
        # if isinstance(self, wx.Frame):
        #     #run this if only we are the main frame
        #     #self.SetTitle('PYME Visualise - ' + filename)
        #     self._removeOldTabs()
        #     self._createNewTabs()
        #
        #     self.CreateFoldPanel()
        #     logger.debug('Gui stuff done')
        
        self.update_datasource_panel()
    
        if recipe_callback:
            recipe_callback()
    
        self.SetFit()
    
        #wx.CallLater(100, self._create_base_layer)
        #wx.CallAfter(self.RefreshView)
        
