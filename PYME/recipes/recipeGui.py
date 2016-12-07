# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:33:47 2015

@author: david
"""

import wx
import wx.gizmos
import wx.html

import numpy as np

import inspect
import locale

#fudge to make things load properly under wx
locale.setlocale(locale.LC_CTYPE, 'C')
import docutils.core

from PYME.recipes import modules
#from PYME.recipes import runRecipe
from PYME.recipes import batchProcess
from PYME.recipes import recipeLayout

import pylab
from PYME.IO.image import ImageStack
from PYME.DSView import ViewIm3D

from PYME.contrib import wxPlotPanel


# try:
#     from traitsui.api import Controller
# except SystemExit:
#     def Controller(*args, **kwargs):
#         """Spoofed traitsui Controller"""
#         pass

import os
import glob
import textwrap

import logging
logger = logging.getLogger(__name__)


RECIPE_DIR = os.path.join(os.path.split(modules.__file__)[0], 'Recipes')
CANNED_RECIPES = glob.glob(os.path.join(RECIPE_DIR, '*.yaml'))

    
class RecipePlotPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, recipes, **kwargs):
        self.recipes = recipes
        self.parent = parent

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )
        
        self.figure.canvas.mpl_connect('pick_event', self.parent.OnPick)

    def draw(self):
        if not self.IsShownOnScreen():
            return
            
        if not hasattr( self, 'ax' ):
            self.ax = self.figure.add_axes([0, 0, 1, 1])

        self.ax.cla()

        dg = self.recipes.activeRecipe.dependancyGraph()

        #Find the connecting lines
        node_positions, connecting_lines = recipeLayout.layout(dg)

        
        axisWidth = self.ax.get_window_extent().width
        nCols = max([1] + [v[0] for v in node_positions.values()])
        pix_per_col = axisWidth/float(nCols)
        
        fontSize = max(6, min(10, 10*pix_per_col/100.))
        
        print pix_per_col, fontSize
        
        TW = textwrap.TextWrapper(width=int(1.8*pix_per_col/fontSize), subsequent_indent='  ')
        TW2 = textwrap.TextWrapper(width=int(1.3*pix_per_col/fontSize), subsequent_indent='  ')
    
        cols = {}

        #Plot the connecting lines
        for xv, yv, e in connecting_lines:
            #choose a colour at random for this input
            if not e in cols.keys():
                cols[e] = 0.7 * np.array(pylab.cm.hsv(pylab.rand()))

            self.ax.plot(xv, yv, c=cols[e], lw=2)
                
        #plot the boxes and the labels
        for k, v in node_positions.items():
            if not (isinstance(k, str) or isinstance(k, unicode)):
                #node - draw a box
                #################
                s = k.__class__.__name__
                #pylab.plot(v[0], v[1], 'o', ms=5)
                if isinstance(k, modules.base.OutputModule):
                    fc = [.8, 1, .8]
                else:
                    fc = [.8,.8, 1]
                rect = pylab.Rectangle([v[0], v[1]-.25], 1, .5, ec='k', fc=fc, picker=True)
                
                rect._data = k
                self.ax.add_patch(rect)
                s = TW2.wrap(s)
                if len(s) == 1:
                    self.ax.text(v[0]+.05, v[1]+.18 , s[0], size=fontSize, weight='bold')
                else:
                    self.ax.text(v[0]+.05, v[1]+.18 - .05*(len(s) - 1) , '\n'.join(s), size=fontSize, weight='bold')
                #print repr(k)
                
                params = k.get().items()
                s2 = []
                for i in params:
                    pn, p = i
                    if not (pn.startswith('_') or pn.startswith('input') or pn.startswith('output')):
                        s2 += TW.wrap('%s : %s' %i)
                
                if len(s2) > 5:
                    s2 = '\n'.join(s2[:4]) + '\n ...'
                else:
                    s2 = '\n'.join(s2)
                self.ax.text(v[0]+.05, v[1]-.22 , s2, size=.8*fontSize, stretch='ultra-condensed')
            else:
                #line - draw an output dot, and a text label 
                s = k
                if not k in cols.keys():
                    cols[k] = 0.7*np.array(pylab.cm.hsv(pylab.rand()))
                self.ax.plot(v[0], v[1], 'o', color=cols[k])
                if k.startswith('out'):
                    t = self.ax.text(v[0]+.1, v[1] + .02, s, color=cols[k], size=fontSize, weight='bold', picker=True, bbox={'color':'w','edgecolor':'k'})
                else:
                    t = self.ax.text(v[0]+.1, v[1] + .02, s, color=cols[k], size=fontSize, weight='bold', picker=True)
                t._data = k
                
                
                
        ipsv = np.array(node_positions.values())
        try:
            xmn, ymn = ipsv.min(0)
            xmx, ymx = ipsv.max(0)
        
            self.ax.set_ylim(ymn-1, ymx+1)
            self.ax.set_xlim(xmn-.5, xmx + .7)
        except ValueError:
            pass
        
        self.ax.axis('off')
        self.ax.grid()
        
        self.canvas.draw()


class ModuleSelectionDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Select a module to add", size=(1000, 500))

        self.pan = wx.Panel(self)

        modNames = modules.base.all_modules.keys()
        modNames.sort()

        self.rootNodes = {}
        self.modnames = {}

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tree_list = wx.gizmos.TreeListCtrl(self.pan, -1, size=(250, 400), style=wx.TR_DEFAULT_STYLE|wx.TR_HIDE_ROOT|wx.TR_FULL_ROW_HIGHLIGHT|
                                                            wx.TR_LINES_AT_ROOT)

        self.tree_list.AddColumn('Module name')
        #self.tree_list.AddColumn('Description')
        self.tree_list.SetMainColumn(0)
        self.tree_list.SetColumnWidth(0, 250)
        #self.tree_list.SetColumnWidth(1, 600)

        root = self.tree_list.AddRoot('root')
        self.tree_list.SetItemText(root, "root", 0)

        for mn in modNames:
            basename, modname = mn.split('.')
            try:
                base = self.rootNodes[basename]
            except KeyError:
                base = self.tree_list.AppendItem(root, basename)
                self.tree_list.SetItemText(base, basename, 0)
                self.rootNodes[basename] = base

            item = self.tree_list.AppendItem(base, modname)
            self.tree_list.SetPyData(item, mn)
            self.tree_list.SetItemText(item, modname, 0)
            #try:
            #    doc = inspect.getdoc(modules.base.all_modules[mn]).splitlines()[0]
            #    self.tree_list.SetItemText(item,doc, 1)
            #except AttributeError:
            #    pass

            #self.modnames[item] = mn

        self.tree_list.ExpandAll(root)

        #self.tree_list.GetMainWindow().Bind(wx.EVT_LEFT_UP, self.OnSelect)
        self.tree_list.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelect)

        hsizer.Add(self.tree_list, 1, wx.EXPAND|wx.ALL, 2)

        self.stModuleHelp = wx.html.HtmlWindow(self, -1, size=(400, -1))#wx.StaticText(self, -1, '', size=(400, -1))
        hsizer.Add(self.stModuleHelp, 0, wx.EXPAND|wx.ALL, 5)

        vsizer.Add(hsizer, 1, wx.EXPAND|wx.ALL, 0)

        sbsizer = wx.StdDialogButtonSizer()

        self.bOK = wx.Button(self.pan, wx.ID_OK, 'Add')
        self.bOK.Enable(False)
        #self.bOK.Bind(wx.EVT_BUTTON, self.OnOK)

        sbsizer.AddButton(self.bOK)

        self.bCancel = wx.Button(self.pan, wx.ID_CANCEL, 'Cancel')
        sbsizer.AddButton(self.bCancel)

        sbsizer.Realize()

        vsizer.Add(sbsizer, 0, wx.EXPAND, 0)

        self.pan.SetSizerAndFit(vsizer)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.pan, 1, wx.EXPAND, 0)
        self.SetSizerAndFit(sizer)
        self.Layout()

    def OnSelect(self, evt):
        from sphinx.util.docstrings import prepare_docstring


        #print self.tree_list.GetSelection()
        mn = self.tree_list.GetPyData(self.tree_list.GetSelection())
        #print mn
        if not mn is None:
            self.bOK.Enable(True)

            doc = modules.base.all_modules[mn].__doc__
            if doc:
                doc = [mn, '#'*len(mn), ''] + prepare_docstring(doc)
                docHTML = docutils.core.publish_parts('\n'.join(doc), writer_name='html')['html_body']
                #print docHTML
                self.stModuleHelp.SetPage(docHTML)
            else:
                self.stModuleHelp.SetPage('')
            #self.stModuleHelp.SetLabelText(doc)
            #self.stModuleHelp.Wrap(self.stModuleHelp.GetSize()[0] - 20)
        else:
            self.stModuleHelp.SetPage('')
            self.bOK.Enable(False)


    def GetSelectedModule(self):
        return self.tree_list.GetPyData(self.tree_list.GetSelection())


class RecipeView(wx.Panel):
    def __init__(self, parent, recipes):
        wx.Panel.__init__(self, parent, size=(400, 100))
        
        self.recipes = recipes
        recipes.recipeView = self
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
        
        self.recipes.LoadRecipeText('')
        
    def update(self):
        self.recipePlot.draw()
        self.tRecipeText.SetValue(self.recipes.activeRecipe.toYAML())
        
    def OnApplyText(self, event):
        self.recipes.LoadRecipeText(self.tRecipeText.GetValue())
        
    def OnNewRecipe(self, event):
        self.recipes.LoadRecipeText('')
        
    def OnAddModule(self, event):
        #mods = 
        mods = modules.base.all_modules
        #modNames = mods.keys()
        #modNames.sort()
        #
        # dlg = wx.SingleChoiceDialog(
        #         self, 'Select module to add', 'Add a module',
        #         modNames,
        #         wx.CHOICEDLG_STYLE
        #         )
        #
        # if dlg.ShowModal() == wx.ID_OK:
        #     modName = dlg.GetStringSelection()

        dlg = ModuleSelectionDialog(self)
        if dlg.ShowModal() == wx.ID_OK:
            modName = dlg.GetSelectedModule()
            
            c = mods[modName](self.recipes.activeRecipe)
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
                if not 'dsviewer' in dir(self.recipes):
                    dv = ViewIm3D(outp, mode='lite')
                else:
                    if self.recipes.dsviewer.mode == 'visGUI':
                        mode = 'visGUI'
                    else:
                        mode = 'lite'
                                   
                    dv = ViewIm3D(outp, mode=mode, glCanvas=self.recipes.dsviewer.glCanvas)
    
    
    def configureModule(self, k):
        p = self
        from traitsui.api import Controller
        class MControl(Controller):
            def closed(self, info, is_ok):
                wx.CallLater(10, p.update)
        
        k.edit_traits(handler=MControl())
        
        
class RecipeManager(object):
    def __init__(self):
        self.activeRecipe = None
        self.LoadRecipeText('')
        
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
        self.activeRecipe = modules.ModuleCollection.fromYAML(s)
        #self.mICurrent.SetItemLabel('Run %s\tF5' % os.path.split(filename)[1])

        try:        
            self.recipeView.update()
        except AttributeError:
            pass


class dt(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window
        
    def OnDropFiles(self, x, y, filenames):
        self.window.UpdateFileList(filenames)
    

class BatchFrame(wx.Frame, wx.FileDropTarget):
    def __init__(self, parent=None):                
        wx.Frame.__init__(self, parent, wx.ID_ANY, 'The PYME Bakery')
        
        self.dropFiles = dt(self)
        logger.debug('BatchFrame.__init__ start')
        self.rm = RecipeManager()
        self.inputFiles = []
        self.inputFiles2 = []
        
        vsizer1=wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, "Recipe:"), wx.HORIZONTAL)
        self.recipeView = RecipeView(self, self.rm)
        
        hsizer.Add(self.recipeView, 1, wx.ALL|wx.EXPAND, 2)
        vsizer1.Add(hsizer, 1, wx.ALL|wx.EXPAND, 2)
        
        hsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        
        vsizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Input files:'), wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Filename pattern:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tGlob = wx.TextCtrl(self, -1, '')
        hsizer.Add(self.tGlob, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.bLoadFromGlob = wx.Button(self, -1, 'Get Matches')
        self.bLoadFromGlob.Bind(wx.EVT_BUTTON, self.OnGetMatches)
        hsizer.Add(self.bLoadFromGlob, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        vsizer2.Add(hsizer, 0, wx.EXPAND, 0)
        
        self.lFiles = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.LC_HRULES)
        self.lFiles.InsertColumn(0, 'Filename')
        self.lFiles.Append(['Either drag files here, or enter a pattern (e.g. /Path/to/data/*.tif ) above and click "Get Matches"',])
        self.lFiles.SetColumnWidth(0, -1)
        
        vsizer2.Add(self.lFiles, .5, wx.EXPAND, 0)        
        
        hsizer1.Add(vsizer2, 0, wx.EXPAND, 10)
        
        vsizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Input files - 2nd channel [optional]:'), wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Filename pattern:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tGlob2 = wx.TextCtrl(self, -1, '')
        hsizer.Add(self.tGlob2, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.bLoadFromGlob2 = wx.Button(self, -1, 'Get Matches')
        self.bLoadFromGlob2.Bind(wx.EVT_BUTTON, self.OnGetMatches2)
        hsizer.Add(self.bLoadFromGlob2, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        vsizer2.Add(hsizer, 0, wx.EXPAND, 0)
        
        self.lFiles2 = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.LC_HRULES)
        self.lFiles2.InsertColumn(0, 'Filename')
        self.lFiles2.Append(['Either drag files here, or enter a pattern (e.g. /Path/to/data/*.tif ) above and click "Get Matches"',])
        self.lFiles2.SetColumnWidth(0, -1)
        
        vsizer2.Add(self.lFiles2, .5, wx.EXPAND, 0)        
        
        hsizer1.Add(vsizer2, 0, wx.EXPAND, 10)
        vsizer1.Add(hsizer1, 0, wx.EXPAND|wx.TOP, 10)
        
        hsizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Output Directory:'), wx.HORIZONTAL)
        
        self.dcOutput = wx.DirPickerCtrl(self, -1, style=wx.DIRP_USE_TEXTCTRL)
        hsizer2.Add(self.dcOutput, 1, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 2)
        
        vsizer1.Add(hsizer2, 0, wx.EXPAND|wx.TOP, 10)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.AddStretchSpacer()

        self.bBake = wx.Button(self, -1, 'Bake') 
        hsizer.Add(self.bBake, 0, wx.ALL, 5)
        self.bBake.Bind(wx.EVT_BUTTON, self.OnBake)
        
        vsizer1.Add(hsizer, 0, wx.EXPAND|wx.TOP, 10)
                
        self.SetSizerAndFit(vsizer1)
        
        #self.SetDropTarget(self.drop)
        self.lFiles.SetDropTarget(self.dropFiles)

        logger.debug('BatchFrame.__init__ done')
        
    def UpdateFileList(self, filenames):
        self.inputFiles += filenames        
        
        self.lFiles.DeleteAllItems()
        
        for f in filenames:
            self.lFiles.Append([f,])
        
    def OnGetMatches(self, event=None):
        import glob
        
        files = glob.glob(self.tGlob.GetValue())
        self.UpdateFileList(files)
        
    def UpdateFileList2(self, filenames):
        self.inputFiles2 += filenames        
        
        self.lFiles2.DeleteAllItems()
        
        for f in filenames:
            self.lFiles2.Append([f,])
        
    def OnGetMatches2(self, event=None):
        import glob
        
        files = glob.glob(self.tGlob2.GetValue())
        self.UpdateFileList2(files)
        
    def OnBake(self, event=None):
        out_dir = self.dcOutput.GetPath()
        
        #validate our choices:
        if (self.rm.activeRecipe is None) or (len(self.rm.activeRecipe.modules) == 0):
            wx.MessageBox('No Recipe: Please open (or build) a recipe', 'Error', wx.OK|wx.ICON_ERROR)
            return
            
        if not len(self.inputFiles) > 0:
            wx.MessageBox('No input files', 'Error', wx.OK|wx.ICON_ERROR)
            return
            
        if (out_dir == '') or not os.path.exists(out_dir):
            wx.MessageBox('Ouput directory does not exist', 'Error', wx.OK|wx.ICON_ERROR)
            return
        
        if not len(self.inputFiles) == len(self.inputFiles2):            
            batchProcess.bake(self.rm.activeRecipe, {'input':self.inputFiles}, out_dir)
        else:
            batchProcess.bake(self.rm.activeRecipe, {'input':self.inputFiles, 'input2':self.inputFiles2}, out_dir)
        
            
   
