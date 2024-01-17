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
import six

#from PYME.recipes import modules
from PYME.recipes import recipeGui
from PYME.IO.image import ImageStack
from PYME.DSView import ViewIm3D
from PYME.DSView import overlays
from PYME.LMVis import pipeline

import os

from ._base import Plugin
        

class RecipePlugin(recipeGui.PipelineRecipeManager, Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        if not 'pipeline' in dir(dsviewer):
            dsviewer.pipeline = pipeline.Pipeline(execute_on_invalidation=False)
        
        recipeGui.PipelineRecipeManager.__init__(self, dsviewer.pipeline)
        
        self.cannedIDs = {}

        dsviewer.AddMenuItem('Recipes', "Load Recipe", self.OnLoadRecipe)
        self.mICurrent = dsviewer.AddMenuItem('Recipes', "Run Current Recipe\tF5", self.RunCurrentRecipe)
        self.mITestCurrent = dsviewer.AddMenuItem('Recipes', "Test Current Recipe\tF7", self.TestCurrentRecipe)
        self.mICurrent = dsviewer.AddMenuItem('Recipes', "Run Current Recipe and Save\tShift+F5", self.RunCurrentRecipeAndSave)
        
        #print CANNED_RECIPES
        
        if len(recipeGui.CANNED_RECIPES) > 0:
            dsviewer.AddMenuItem('Recipes', '', itemType='separator')
            #self.mRecipes.AppendSeparator()
            
            for r in recipeGui.CANNED_RECIPES:
                #print r, 
                ID = dsviewer.AddMenuItem('Recipes', os.path.split(r)[1], self.OnRunCanned).GetId()
                #ID = wx.NewIdRef()
                self.cannedIDs[ID] = r
                #self.mRecipes.Append(ID, os.path.split(r)[1],"",  wx.ITEM_NORMAL)
                #wx.EVT_MENU(dsviewer, ID, self.OnRunCanned)

        # custom recipes - load only, do not execute
        import PYME.config
        customRecipes = PYME.config.get_custom_recipes()
        if len(customRecipes) > 0:
            dsviewer.AddMenuItem('Recipes', '', itemType='separator')
            for r in customRecipes:
                ID = dsviewer.AddMenuItem('Recipes', r, self.OnLoadCustom).GetId()
                self.cannedIDs[ID] = customRecipes[r]

        #dsviewer.menubar.Append(self.mRecipes, "Recipes")
        dsviewer.AddMenuItem('Recipes', '', itemType='separator')
        dsviewer.AddMenuItem('Recipes', "Save Results", self.OnSaveOutputs)
        dsviewer.AddMenuItem('Recipes', "Save Results - Old Style", self.OnSaveOutputOld)
        
        #dsviewer.AddMenuItem('Recipes', '', itemType='separator')
        dsviewer.AddMenuItem('Recipes', "Load Previous Results", self.OnLoadOutputs)
            
        self.recipeView = recipeGui.RecipeView(dsviewer, self)
        dsviewer.AddPage(page=self.recipeView, select=False, caption='Recipe')
        
    def RunCurrentRecipeAndSave(self, event=None):
        self.RunCurrentRecipe(saveResults=True)
        
    def RunCurrentRecipe(self, event=None, testMode=False, saveResults = False):
        if self.activeRecipe:
            if testMode:
                #just run on current frame  # FIXME - this breaks SUPERTILE datasources and anything that needs to carry datasource attributes forward
                self.outp = self.activeRecipe.execute(input=ImageStack([np.atleast_3d(self.image.data[:,:,self.do.zp, c]) for c in range(self.image.data.shape[3])], mdh=self.image.mdh))
            else:
                #run normally
                self.outp = self.activeRecipe.execute(input=self.image)
                
                if saveResults:
                    dir_dialog = wx.DirDialog(None, 'Set output directory', style=wx.FD_OPEN)
                    succ = dir_dialog.ShowModal()
                    if (succ == wx.ID_OK):
                        output_dir = dir_dialog.GetPath()
                        file_stub = os.path.splitext(os.path.split(self.image.filename)[-1])[0]
                        self.activeRecipe.save({'output_dir': output_dir, 
                                                'file_stub': file_stub})
                    
                    
            def _display_output_image(outp):
                if self.dsviewer.mode == 'visGUI':
                    mode = 'visGUI'
                elif 'out_tracks' in self.activeRecipe.namespace.keys():
                    mode = 'tracking'
                else:
                    mode = 'default'
    
                dv = ViewIm3D(outp, mode=mode, glCanvas=self.dsviewer.glCanvas)
    
                if 'out_meas' in self.activeRecipe.namespace.keys():
                    #have measurements as well - add to / overlay with output image
                    if not 'pipeline' in dir(dv):
                        dv.pipeline = pipeline.Pipeline()
        
                    from PYME.IO import tabular
                    cache = tabular.CachingResultsFilter(self.activeRecipe.namespace['out_meas'])
                    
                    dv.pipeline.OpenFile(ds=cache) # TODO - is needed?
                    dv.view.add_overlay(overlays.PointDisplayOverlay(filter=cache, display_name='out_meas'))
                    
                    #dv.view.filter = dv.pipeline
    
                #set scaling to (0,1)
                for i in range(outp.data.shape[3]):
                    dv.do.Gains[i] = 1.0
    
                if ('out_tracks' in self.activeRecipe.namespace.keys()) and 'tracker' in dir(dv):
                    dv.tracker.SetTracks(self.activeRecipe.namespace['out_tracks'])
                    
            def _display_output_report(outp):
                import wx.html2
                html_view = wx.html2.WebView.New(self.dsviewer)
                self.dsviewer.AddPage(html_view, True, 'Recipe Report')
                html_view.SetPage(outp, 'Recipe Report')

            if ('out_tracks' in self.activeRecipe.namespace.keys()) and 'tracker' in dir(self.dsviewer):
                self.dsviewer.tracker.SetTracks(self.activeRecipe.namespace['out_tracks'])

            #assume we made measurements - put in pipeline
            #TODO - put an explict check in here

            

            if isinstance(self.outp, ImageStack):
                _display_output_image(self.outp)
            elif not self.outp is None:
                #from PYME.IO import tabular
                
                
                #cache = tabular.CachingResultsFilter(self.outp)
                #self.dsviewer.pipeline.OpenFile(ds = cache, clobber_recipe=False)
                #self.dsviewer.pipeline.filterKeys = {}
                #self.dsviewer.pipeline.Rebuild()

                if not hasattr(self, '_ovl'):
                    self._ovl = overlays.PointDisplayOverlay(filter=self.dsviewer.pipeline, display_name='Recipe output')
                    self.view.add_overlay(self.ovl)
                    
                
            for out_ in self.activeRecipe.gather_outputs():
                if isinstance(out_, ImageStack):
                    _display_output_image(out_)
                elif isinstance(out_, six.string_types):
                    _display_output_report(out_)
                    
                

                
    def TestCurrentRecipe(self, event=None):
        """run recipe on current frame only as an inexpensive form of testing"""
        
        self.RunCurrentRecipe(testMode=True)
                
    def OnRunCanned(self, event):
        self.LoadRecipe(self.cannedIDs[event.GetId()])
        self.RunCurrentRecipe()
        
    def OnLoadCustom(self, event):
        self.LoadRecipe(self.cannedIDs[event.GetId()])

    def OnSaveOutputs(self, event):
        self.activeRecipe.save()
        # from PYME.recipes import runRecipe
        #
        # filename = wx.FileSelector('Save results as ...',
        #                            wildcard="CSV files (*.csv)|*.csv|Excell files (*.xlsx)|*.xlsx|HDF5 files (*.hdf)|*.hdf",
        #                            flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        #
        # if not filename == '':
        #     runRecipe.saveOutput(self.outp, filename)
            
        
            
        self.activeRecipe.save()

    def OnSaveOutputOld(self, event):
        from PYME.recipes import runRecipe
        
        filename = wx.FileSelector('Save results as ...',
                                   wildcard="CSV files (*.csv)|*.csv|Excell files (*.xlsx)|*.xlsx|HDF5 files (*.hdf)|*.hdf",
                                   flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
                                   
        if not filename == '':
            runRecipe.saveOutput(self.outp, filename)


    def OnLoadOutputs(self, event):
        import pandas
        from PYME.IO import tabular
        
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
                
            cache = tabular.CachingResultsFilter(data)
            self.dsviewer.pipeline.OpenFile(ds = cache)

            if not hasattr(self, '_ovl'):
                self._ovl = overlays.PointDisplayOverlay(filter=self.dsviewer.pipeline, display_name=filename)
                self.view.add_overlay(self.ovl)
                
                
            
        



        




def Plug(dsviewer):
    # dsviewer.create_overlay_panel()
    return RecipePlugin(dsviewer)
    




