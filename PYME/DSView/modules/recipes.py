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
import numpy as np
import wx
from PYME.DSView.OverlaysPanel import OverlayPanel
import wx.lib.agw.aui as aui

from PYME.Analysis.Modules import filters
import pylab
from PYME.DSView.image import ImageStack
from PYME.DSView import ViewIm3D

from PYME.Analysis.LMVis import pipeline
from PYME.misc import wxPlotPanel
from PYME.misc import depGraph

from traitsui.api import Controller

import os
import glob

RECIPE_DIR = os.path.join(os.path.split(filters.__file__)[0], 'Recipes')
CANNED_RECIPES = glob.glob(os.path.join(RECIPE_DIR, '*.yaml'))


class RecipePlotPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, recipes, **kwargs):
        self.recipes = recipes
        self.parent = parent

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )
        
        self.figure.canvas.mpl_connect('pick_event', self.parent.OnPick)

    def draw(self):
        if not hasattr( self, 'ax' ):
            self.ax = self.figure.add_axes([0, 0, 1, 1])

        self.ax.cla()

        dg = self.recipes.activeRecipe.dependancyGraph()        
        ips = depGraph.arrangeNodes(dg)
    
    
        cols = {}    
        for k, v in dg.items():
            if not (isinstance(k, str) or isinstance(k, unicode)):
                yv0 = []
                yoff = .1*np.arange(len(v))
                yoff -= yoff.mean()
                
                for e in v:
                    x0, y0 = ips[e]
                    x1, y1 = ips[k]
                    yv0.append(y0 + 0.01*x0*(2.0*(y0>y1) - 1))
                    
                yvi = np.argsort(np.array(yv0))
                #print yv0, yvi
                yos = np.zeros(3)
                yos[yvi] = yoff
                    
                for e, yo in zip(v, yos):
                    x0, y0 = ips[e]
                    x1, y1 = ips[k]
                    
                    if not e in cols.keys():
                        cols[e] = 0.7*np.array(pylab.cm.hsv(pylab.rand()))
                    
                    self.ax.plot([x0,x0+.5, x0+.5, x1], [y0,y0,y1+yo,y1+yo], c=cols[e], lw=2)
                
        for k, v in ips.items():   
            if not (isinstance(k, str) or isinstance(k, unicode)):
                s = k.__class__.__name__
                #pylab.plot(v[0], v[1], 'o', ms=5)
                rect = pylab.Rectangle([v[0], v[1]-.25], 1, .5, ec='k', fc=[.8,.8, 1], picker=True)
                
                rect._data = k
                self.ax.add_patch(rect)
                self.ax.text(v[0]+.05, v[1]+.18 , s, weight='bold')
                #print repr(k)
                
                s2 = '\n'.join(['%s : %s' %i for i in k.get().items()])
                self.ax.text(v[0]+.05, v[1]-.22 , s2, size=8, stretch='ultra-condensed')
            else:
                s = k
                if not k in cols.keys():
                    cols[k] = 0.7*np.array(pylab.cm.hsv(pylab.rand()))
                self.ax.plot(v[0], v[1], 'o', color=cols[k])
                t = self.ax.text(v[0]+.1, v[1] + .02, s, color=cols[k], weight='bold', picker=True)
                t._data = k
                
                
                
        ipsv = np.array(ips.values())
        try:
            xmn, ymn = ipsv.min(0)
            xmx, ymx = ipsv.max(0)
        
            self.ax.set_ylim(ymn-1, ymx+1)
            self.ax.set_xlim(xmn-.5, xmx + .7)
        except ValueError:
            pass
        
        self.ax.axis('off')
        
        self.canvas.draw()




class RecipeView(wx.Panel):
    def __init__(self, parent, recipes):
        wx.Panel.__init__(self, parent, size=(400, 100))
        
        self.recipes = recipes
        hsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.recipePlot = RecipePlotPanel(self, recipes, size=(-1, 400))
        vsizer.Add(self.recipePlot, 1, wx.ALL|wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.bNewRecipe = wx.Button(self, -1, 'New Recipe')
        hsizer.Add(self.bNewRecipe, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bNewRecipe.Bind(wx.EVT_BUTTON, self.OnNewRecipe)

        self.bLoadRecipe = wx.Button(self, -1, 'Load Recipe')
        hsizer.Add(self.bLoadRecipe, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bLoadRecipe.Bind(wx.EVT_BUTTON, self.recipes.OnLoadRecipe)
        
        self.bAddModule = wx.Button(self, -1, 'Add Module')
        hsizer.Add(self.bAddModule, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bAddModule.Bind(wx.EVT_BUTTON, self.OnAddModule)
        
        #self.bRefresh = wx.Button(self, -1, 'Refresh')
        #hsizer.Add(self.bRefresh, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.bSaveRecipe = wx.Button(self, -1, 'Save Recipe')
        hsizer.Add(self.bSaveRecipe, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSaveRecipe.Bind(wx.EVT_BUTTON, self.recipes.OnSaveRecipe)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        hsizer1.Add(vsizer, 1, wx.EXPAND|wx.ALL, 2)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.tRecipeText = wx.TextCtrl(self, -1, '', size=(250, -1),
                                       style=wx.TE_MULTILINE|wx.TE_PROCESS_ENTER)
                                       
        vsizer.Add(self.tRecipeText, 1, wx.ALL, 2)
        
        self.bApply = wx.Button(self, -1, 'Apply')
        vsizer.Add(self.bApply, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bApply.Bind(wx.EVT_BUTTON, self.OnApplyText)
                                       
        hsizer1.Add(vsizer, 0, wx.EXPAND|wx.ALL, 2)

                
        
        self.SetSizerAndFit(hsizer1)
        
    def update(self):
        self.recipePlot.draw()
        self.tRecipeText.SetValue(self.recipes.activeRecipe.toYAML())
        
    def OnApplyText(self, event):
        self.recipes.LoadRecipeText(self.tRecipeText.GetValue())
        
    def OnNewRecipe(self, event):
        self.recipes.LoadRecipeText('')
        
    def OnAddModule(self, event):
        mods = [c for c in filters.__dict__.values() if filters._issubclass(c, filters.ModuleBase) and not c == filters.ModuleBase]
        modNames = [c.__name__ for c in mods]        
        
        dlg = wx.SingleChoiceDialog(
                self, 'Select module to add', 'Add a module',
                modNames, 
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            modName = dlg.GetStringSelection()
            
            c = mods[modNames.index(modName)]()
            self.recipes.activeRecipe.modules.append(c)
        dlg.Destroy()
        
        self.configureModule(c)
        
    def OnPick(self, event):
        k = event.artist._data
        if not (isinstance(k, str) or isinstance(k, unicode)):
            self.configureModule(k)
        else:
            outp = self.recipes.activeRecipe.namespace[k]
            if isinstance(outp, ImageStack):
                if self.recipes.dsviewer.mode == 'visGUI':
                    mode = 'visGUI'
                else:
                    mode = 'lite'
    
                dv = ViewIm3D(outp, mode=mode, glCanvas=self.recipes.dsviewer.glCanvas)
    
    
    def configureModule(self, k):
        p = self
        class MControl(Controller):
            def closed(self, info, is_ok):
                wx.CallLater(10, p.update)
        
        k.edit_traits(handler=MControl())
        
        
        

class Recipes:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        self.activeRecipe = None
        
        self.cannedIDs = {}
        
#        self.mRecipes = wx.Menu()
#        
#        LOAD_RECIPE = wx.NewId()
#        RUN_CURRENT = wx.NewId()
#        
#        
#        self.mRecipes.Append(LOAD_RECIPE, "Load Recipe", "", wx.ITEM_NORMAL)
#        self.mICurrent = self.mRecipes.Append(RUN_CURRENT, "Run Current Recipe\tF5", "", wx.ITEM_NORMAL)
#    
#        wx.EVT_MENU(dsviewer, LOAD_RECIPE, self.OnLoadRecipe)
#        wx.EVT_MENU(dsviewer, RUN_CURRENT, self.RunCurrentRecipe)
        
        dsviewer.AddMenuItem('Recipes', "Load Recipe", self.OnLoadRecipe)
        self.mICurrent = dsviewer.AddMenuItem('Recipes', "Run Current Recipe\tF5", self.RunCurrentRecipe)
        
        #print CANNED_RECIPES
        
        if len(CANNED_RECIPES) > 0:
            dsviewer.AddMenuItem('Recipes', '', itemType='separator')
            #self.mRecipes.AppendSeparator()
            
            for r in CANNED_RECIPES:
                #print r, 
                ID = dsviewer.AddMenuItem('Recipes', os.path.split(r)[1], self.OnRunCanned).GetId()
                #ID = wx.NewId()
                self.cannedIDs[ID] = r
                #self.mRecipes.Append(ID, os.path.split(r)[1],"",  wx.ITEM_NORMAL)
                #wx.EVT_MENU(dsviewer, ID, self.OnRunCanned)
            
        #dsviewer.menubar.Append(self.mRecipes, "Recipes")
            
        self.recipeView = RecipeView(dsviewer, self)
        dsviewer.AddPage(page=self.recipeView, select=False, caption='Recipe')
        
        

    def OnLoadRecipe(self, event):
        filename = wx.FileSelector("Choose a recipe to open",  
                                   default_extension='yaml', 
                                   wildcard='PYME Recipes (*.yaml)|*.yaml')

        #print filename
        if not filename == '':
            self.LoadRecipe(filename)
            
    def OnSaveRecipe(self, event):
        filename = wx.FileSelector("Save Recipe as ... ",  
                                   default_extension='yaml', 
                                   wildcard='PYME Recipes (*.yaml)|*.yaml', flags=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        #print filename
        if not filename == '':
            s = self.activeRecipe.toYAML()
            with open(filename, 'w') as f:
                f.write(s)
            

    def LoadRecipe(self, filename):
        #self.currentFilename  = filename
        with open(filename) as f:
            s = f.read()
        
        self.LoadRecipeText(s, filename)
            
    def LoadRecipeText(self, s, filename=''):
        self.currentFilename  = filename
        self.activeRecipe = filters.ModuleCollection.fromYAML(s)
        self.mICurrent.SetItemLabel('Run %s\tF5' % os.path.split(filename)[1])
        self.recipeView.update()
    
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
                self.dsviewer.view.filter = self.dsviewer.pipeline
                
    def OnRunCanned(self, event):
        self.LoadRecipe(self.cannedIDs[event.GetId()])
        self.RunCurrentRecipe()

        




def Plug(dsviewer):
    dsviewer.recipes = Recipes(dsviewer)
    
    if not 'overlaypanel' in dir(dsviewer):    
        dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
        dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
        pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)
    
        dsviewer.panesToMinimise.append(pinfo2)



