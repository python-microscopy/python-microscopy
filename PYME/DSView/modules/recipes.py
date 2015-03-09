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
#import numpy
import wx
from PYME.Analysis.Modules import filters
#import pylab
from PYME.DSView.image import ImageStack
from PYME.DSView import ViewIm3D

from PYME.Analysis.LMVis import pipeline

import os
import glob

RECIPE_DIR = os.path.join(os.path.split(filters.__file__)[0], 'Recipes')
CANNED_RECIPES = glob.glob(os.path.join(RECIPE_DIR, '*.yaml'))

class Recipes:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        self.activeRecipe = None
        
        self.cannedIDs = {}
        
        #self.mRecipes = wx.Menu()
        
        LOAD_RECIPE = wx.NewId()
        RUN_CURRENT = wx.NewId()
        
        dsviewer.mProcessing.AppendSeparator()
        dsviewer.mProcessing.Append(LOAD_RECIPE, "Load Recipe", "", wx.ITEM_NORMAL)
        self.mICurrent = dsviewer.mProcessing.Append(RUN_CURRENT, "Run Current Recipe", "", wx.ITEM_NORMAL)
    
        wx.EVT_MENU(dsviewer, LOAD_RECIPE, self.OnLoadRecipe)
        wx.EVT_MENU(dsviewer, RUN_CURRENT, self.RunCurrentRecipe)
        
        #print CANNED_RECIPES
        
        if len(CANNED_RECIPES) > 0:
            self.mRecipes = wx.Menu()
            for r in CANNED_RECIPES:
                #print r, 
                ID = wx.NewId()
                self.cannedIDs[ID] = r
                self.mRecipes.Append(ID, os.path.split(r)[1],"",  wx.ITEM_NORMAL)
                wx.EVT_MENU(dsviewer, ID, self.OnRunCanned)
            
            dsviewer.mProcessing.AppendSubMenu(self.mRecipes, "Run Canned Recipe")
        
        

    def OnLoadRecipe(self, event):
        filename = wx.FileSelector("Choose a recipe to open",  
                                   default_extension='yaml', 
                                   wildcard='PYME Recipes (*.yaml)')

        #print filename
        if not filename == '':
            self.LoadRecipe(filename)
            

    def LoadRecipe(self, filename):
        self.currentFilename  = filename
        with open(filename) as f:
            s = f.read()
        
        self.activeRecipe = filters.ModuleCollection.fromYAML(s)
        self.mICurrent.SetItemLabel('Run %s' % os.path.split(filename)[1])
    
    def RunCurrentRecipe(self, event=None):
        if self.activeRecipe:
            outp = self.activeRecipe.execute(input=self.image)
            if isinstance(outp, ImageStack):
                if self.dsviewer.mode == 'visGUI':
                    mode = 'visGUI'
                else:
                    mode = 'lite'
    
                dv = ViewIm3D(outp, mode=mode, glCanvas=self.dsviewer.glCanvas)
    
                #set scaling to (0,1)
                for i in range(outp.data.shape[3]):
                    dv.do.Gains[i] = 1.0
                    
            else:
                #assume we made measurements - put in pipeline
                #TODO - put an explict check in here
            
                if not 'pipeline' in dir(self.dsviewer):
                    self.dsviewer.pipeline = pipeline.Pipeline()
                    
                self.dsviewer.pipeline.OpenFile(ds=outp)
                
    def OnRunCanned(self, event):
        self.LoadRecipe(self.cannedIDs[event.GetId()])
        self.RunCurrentRecipe()

        




def Plug(dsviewer):
    dsviewer.recipes = Recipes(dsviewer)



