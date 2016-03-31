#!/usr/bin/python
##################
# coloc.py
#
# Copyright David Baddeley, 2011
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
#import numpy as np
#import wx
from PYME.DSView.OverlaysPanel import OverlayPanel
import wx.lib.agw.aui as aui
import wx
import numpy as np

#from PYME.Analysis.Modules import modules
from PYME.Analysis.Modules import recipeGui
from PYME.DSView.image import ImageStack
from PYME.DSView import ViewIm3D

from PYME.Analysis.LMVis import pipeline

import os
        

class RecipePlugin(recipeGui.RecipeManager):
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        self.cannedIDs = {}
        

        dsviewer.AddMenuItem('Recipes', "Load Recipe", self.OnLoadRecipe)
        self.mICurrent = dsviewer.AddMenuItem('Recipes', "Run Current Recipe\tF5", self.RunCurrentRecipe)
        self.mITestCurrent = dsviewer.AddMenuItem('Recipes', "Test Current Recipe\tF7", self.TestCurrentRecipe)
        
        #print CANNED_RECIPES
        
        if len(recipeGui.CANNED_RECIPES) > 0:
            dsviewer.AddMenuItem('Recipes', '', itemType='separator')
            #self.mRecipes.AppendSeparator()
            
            for r in recipeGui.CANNED_RECIPES:
                #print r, 
                ID = dsviewer.AddMenuItem('Recipes', os.path.split(r)[1], self.OnRunCanned).GetId()
                #ID = wx.NewId()
                self.cannedIDs[ID] = r
                #self.mRecipes.Append(ID, os.path.split(r)[1],"",  wx.ITEM_NORMAL)
                #wx.EVT_MENU(dsviewer, ID, self.OnRunCanned)
            
        #dsviewer.menubar.Append(self.mRecipes, "Recipes")
        dsviewer.AddMenuItem('Recipes', '', itemType='separator')
        dsviewer.AddMenuItem('Recipes', "Save Results", self.OnSaveOutputs)
        
        #dsviewer.AddMenuItem('Recipes', '', itemType='separator')
        dsviewer.AddMenuItem('Recipes', "Load Previous Results", self.OnLoadOutputs)
            
        self.recipeView = recipeGui.RecipeView(dsviewer, self)
        dsviewer.AddPage(page=self.recipeView, select=False, caption='Recipe')
        
    def RunCurrentRecipe(self, event=None, testMode=False):
        if self.activeRecipe:
            if testMode:
                #just run on current frame
                self.outp = self.activeRecipe.execute(input=ImageStack([np.atleast_3d(self.image.data[:,:,self.do.zp, c]) for c in range(self.image.data.shape[3])], mdh=self.image.mdh))
            else:
                #run normally
                self.outp = self.activeRecipe.execute(input=self.image)
                
            if isinstance(self.outp, ImageStack):
                if self.dsviewer.mode == 'visGUI':
                    mode = 'visGUI'
                else:
                    mode = 'default'
    
                dv = ViewIm3D(self.outp, mode=mode, glCanvas=self.dsviewer.glCanvas)
                
                if 'out_meas' in self.activeRecipe.namespace.keys():
                    #have measurements as well - add to / overlay with output image
                    if not 'pipeline' in dir(dv):
                        dv.pipeline = pipeline.Pipeline()
                    
                    from PYME.Analysis.LMVis import inpFilt
                    cache = inpFilt.cachingResultsFilter(self.activeRecipe.namespace['out_meas'])
                    dv.pipeline.OpenFile(ds = cache)
                    dv.view.filter = dv.pipeline
                    
    
                #set scaling to (0,1)
                for i in range(self.outp.data.shape[3]):
                    dv.do.Gains[i] = 1.0
                    
            else:
                #assume we made measurements - put in pipeline
                #TODO - put an explict check in here
            
                if not 'pipeline' in dir(self.dsviewer):
                    self.dsviewer.pipeline = pipeline.Pipeline()
                
                from PYME.Analysis.LMVis import inpFilt
                cache = inpFilt.cachingResultsFilter(self.outp)
                self.dsviewer.pipeline.OpenFile(ds = cache)
                self.dsviewer.view.filter = self.dsviewer.pipeline
                
    def TestCurrentRecipe(self, event=None):
        '''run recipe on current frame only as an inexpensive form of testing'''
        
        self.RunCurrentRecipe(testMode=True)
                
    def OnRunCanned(self, event):
        self.LoadRecipe(self.cannedIDs[event.GetId()])
        self.RunCurrentRecipe()
        
    def OnSaveOutputs(self, event):
        from PYME.Analysis.Modules import runRecipe
        
        filename = wx.FileSelector('Save results as ...', 
                                   wildcard="CSV files (*.csv)|*.csv|Excell files (*.xlsx)|*.xlsx|HDF5 files (*.hdf)|*.hdf", 
                                   flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
                                   
        if not filename == '':
            runRecipe.saveOutput(self.outp, filename)
            
    def OnLoadOutputs(self, event):
        import pandas
        from PYME.Analysis.LMVis import inpFilt
        
        filename = wx.FileSelector('Save results as ...', 
                                   wildcard="CSV files (*.csv)|*.csv|Excell files (*.xlsx)|*.xlsx|HDF5 files (*.hdf)|*.hdf", 
                                   flags = wx.FD_OPEN)
                                   
        if not filename == '':
            if filename.endswith('.csv'):
                data = pandas.read_csv(filename)
            elif filename.endswith('.xlsx'):
                data = pandas.read_excel(filename)
            elif filename.endswith('.hdf'):
                data = pandas.read_hdf(filename)
                
            
            if not 'pipeline' in dir(self.dsviewer):
                self.dsviewer.pipeline = pipeline.Pipeline()
                
            cache = inpFilt.cachingResultsFilter(data)
            self.dsviewer.pipeline.OpenFile(ds = cache)
            self.dsviewer.view.filter = self.dsviewer.pipeline
                
                
            
        



        




def Plug(dsviewer):
    dsviewer.recipes = RecipePlugin(dsviewer)
    
    if not 'overlaypanel' in dir(dsviewer):    
        dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
        dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
        pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)
    
        dsviewer.panesToMinimise.append(pinfo2)



