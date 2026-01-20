# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:14:38 2014

@author: Kenny

#PipeViewer

Addon to VisGUI
View the pipeline in a tabular format. Support sorting and highlighting by similarity.

Global variables:
*****************
VisFr: Reference to the VisGUI MainWindow.
pipeline: Reference to VisGUI pipeline. Reroute to use any arbitrary data. Require Dict-like object.
    A new cached verions is built everytime the table is refreshed.
viewingGrid: Reference to the wx.grid.Grid derived Grid GUI. Uses a derived PyGridTableBase for its base data.
setupPanel: Reference to the setup panel.
accEntryList: List for keyboard shortcuts.

"""

import wx
import wx.grid
import numpy as np
import scipy as sp
from scipy import misc, stats
import collections
from matplotlib import pyplot, colorbar, colorbar, ticker

from PYME.IO import tabular

VisFr = None
pipeline = None
viewingGrid = None
setupPanel = None
accEntryList = []

class PipeViewerPanel(wx.Panel):
    '''Base panel.
    '''
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        
        mainSizer.Add(SetupPanel(self), flag=wx.EXPAND)
        mainSizer.Add(ViewingPanel(self), proportion=1, flag=wx.EXPAND)
        
        self.SetSizer(mainSizer)
        
        self.SetAcceleratorTable(wx.AcceleratorTable(accEntryList))

class SetupPanel(wx.Panel):
    '''Setup panel. Contains all the buttons/textboxes, etc.    
    '''
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        global setupPanel
        setupPanel = self
        mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        
        refreshBtn = wx.Button(self, label="Refresh")
        refreshBtn.Bind(wx.EVT_BUTTON, self.RefreshButtonClicked)
        refreshBtn.SetToolTipString("Pipeline is cached for speed. Refresh when data source has been modified.")
        mainSizer.Add(refreshBtn)
        
#        mainSizer.Add(wx.StaticText(self, label="Pipeline is cached for speed. Refresh when needed."))
        
        mainSizer.AddSpacer(50)
        
        hlPrecisionNumeric = wx.SpinCtrl(self, size=(40, -1), min=1, max=10, initial=PipelineGridBase.GetDefaultHighlighterPrecision())          
        hlPrecisionNumeric.Bind(wx.EVT_SPINCTRL, self.HighlighterPrecisionUpdate)
        hlPrecisionNumeric.SetToolTipString("Set precision of data highlighter.\nRight click to highlight data with similar values")
        mainSizer.Add(hlPrecisionNumeric)        
        mainSizer.Add(wx.StaticText(self, label=" Sig. Fig. ", style=wx.ALIGN_CENTRE_HORIZONTAL), flag=wx.CENTER)
        
        mainSizer.AddSpacer(50)
        jumpToBtn = wx.Button(self, label="Jump to")
        jumpToBtn.Bind(wx.EVT_BUTTON, self.JumpToButtonClicked)
        self.colJumpTarget = wx.Choice(self)
        self.rowJumpTarget = wx.SpinCtrl(self, size=(100, -1))
        mainSizer.Add(jumpToBtn)
        mainSizer.Add(self.rowJumpTarget)
        mainSizer.Add(self.colJumpTarget)
        
        mainSizer.AddSpacer(50)
        histPlotButton = wx.Button(self, label="Hist")
        histPlotButton.Bind(wx.EVT_BUTTON, self.HistPlotButtonClicked)
        histPlotButton.SetToolTipString("Select column(s) and click to plot histograms.")
        mainSizer.Add(histPlotButton)
        hist2dPlotButton = wx.Button(self, label="Hist2D")
        hist2dPlotButton.Bind(wx.EVT_BUTTON, self.Hist2dPlotButtonClicked)
        hist2dPlotButton.SetToolTipString("Select columns and click to plot 2D histograms.")
        mainSizer.Add(hist2dPlotButton)
        scatterPlotButton = wx.Button(self, label="Scatter")
        scatterPlotButton.Bind(wx.EVT_BUTTON, self.ScatterPlotButtonClicked)
        scatterPlotButton.SetToolTipString("Select columns and click to plot scatter graph.")
        mainSizer.Add(scatterPlotButton)
        
        mainSizer.AddSpacer(50)
        addMappingButton = wx.Button(self, label="Add mapping")        
        addMappingButton.SetToolTipString("Add a new mapping to the currently selected data source.")
        mainSizer.Add(addMappingButton)
        keyTextCtrl = wx.TextCtrl(self, value="new key")
        keyTextCtrl.SetToolTipString("Key for the new mapping.")
        mainSizer.Add(keyTextCtrl)
        mappingTextCtrl = wx.TextCtrl(self, value="np.sqrt(x**2+y**2)")        
        mappingTextCtrl.SetToolTipString("New mapping. Refer to other keys directly by name.")
        mainSizer.Add(mappingTextCtrl, proportion=1)
        addMappingButton.Bind(wx.EVT_BUTTON, lambda event, key=keyTextCtrl, mapping=mappingTextCtrl: self.AddMappingButtonClicked(event, key, mapping))
        
        self.SetSizer(mainSizer)       
        
        accEntryList.append(wx.AcceleratorEntry(wx.ACCEL_NORMAL, wx.WXK_F5, refreshBtn.Id))
    
    def RefreshButtonClicked(self, event):
        '''Force rebuild of cache.
        '''
        viewingGrid.Rebuild()
    
    def HighlighterPrecisionUpdate(self, event):
        '''Change highlighter precision. Will clear all previous highlight.
        '''
        viewingGrid.GetTable().SetHighlighterPrecision(event.GetInt())
    
    def JumpToButtonClicked(self, event):
        '''Jump and select target cell.        
        '''
        row, col = self.rowJumpTarget.GetValue(), self.colJumpTarget.GetSelection()
        row -= 1
        viewingGrid.ClearSelection()
        viewingGrid.MakeCellVisible(row, col)
        viewingGrid.SetFocus()
        viewingGrid.SetGridCursor(row, col)
    
    def HistPlotButtonClicked(self, event):
        '''Plot 1D histogram.
        '''
        selectedColsCount = len(viewingGrid.GetSelectedCols())
        if selectedColsCount == 0:
            return
        if selectedColsCount > 20:
            with wx.MessageDialog(self, "You are creating > 20 plots. Are you sure?", "Warning", style=wx.CENTER|wx.YES_NO|wx.NO_DEFAULT|wx.ICON_WARNING) as dlg:
                if dlg.ShowModal() == wx.ID_NO:
                    return
            
        rows = np.floor(np.sqrt(selectedColsCount))
        cols = np.ceil(selectedColsCount/rows)

        fig, axes = pyplot.subplots(nrows=int(rows), ncols=int(cols), squeeze=False)
        
        i = 0
        for col in viewingGrid.GetSelectedCols():
            currentAx = axes.flat[i]
            currentAx.xaxis.get_major_formatter().set_powerlimits((-3, 3))
            currentAx.yaxis.get_major_formatter().set_powerlimits((-3, 3))
            currentAx.hist(viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[col]], bins=16)
            currentAx.set_xlabel(viewingGrid.GetTable().keys[col])
            currentAx.set_ylabel("n")
            i += 1
            
        fig.tight_layout()
    
    def ScatterPlotButtonClicked(self, event):
        '''Plot 2D scatterplots with linear fit (if p < 0.1). Skips data points when too many.
        '''
        selectedCols = viewingGrid.GetSelectedCols()
        if len(selectedCols) < 2:
            return
        plotCount = sp.special.comb(len(selectedCols), 2)
        if plotCount > 20:
            with wx.MessageDialog(self, "You are creating > 20 plots. Are you sure?", "Warning", style=wx.CENTER|wx.YES_NO|wx.NO_DEFAULT|wx.ICON_WARNING) as dlg:
                if dlg.ShowModal() == wx.ID_NO:
                    return
        
        rows = np.floor(np.sqrt(plotCount))
        cols = np.ceil(plotCount/rows)
        fig, axes = pyplot.subplots(nrows=int(rows), ncols=int(cols), squeeze=False)

        x = 0
        for i in range(len(selectedCols)):
            for j in range(i + 1, len(selectedCols)):
                currentAx = axes.flat[x]
                currentAx.xaxis.get_major_formatter().set_powerlimits((-3, 3))
                currentAx.yaxis.get_major_formatter().set_powerlimits((-3, 3))

                plotEvery = int(np.ceil(0.0001*viewingGrid.GetTable().rowCount))
                currentAx.plot(viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[i]]],
                                viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[j]]],
                                linestyle='',
                                marker='o',
                                markersize=2,
                                markevery=plotEvery)                
                if not plotEvery == 1:
                    currentAx.annotate("show every {}".format(plotEvery), (0.1, 0.85), xycoords="axes fraction", backgroundcolor='w', axes=currentAx)

                xedges = [viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[i]]].min(),
                          viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[i]]].max()]
                
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[i]]],viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[j]]])
#                print slope, intercept, r_value, p_value, std_err
                if True or p_value < 0.1:
                    fitY=[slope*xedges[0]+intercept, slope*xedges[-1]+intercept]
                    currentAx.autoscale(False)
                    # currentAx.plot(xedges, fitY, linestyle='-', linewidth=3, color="w")
                    currentAx.plot(xedges, fitY, linestyle='--', linewidth=1, color="r", label="Linear Fit")
                    currentAx.annotate("r={:.3f}".format(r_value), (0.1, 0.9), xycoords="axes fraction", backgroundcolor='w', axes=currentAx)
                    currentAx.autoscale(True)

                currentAx.set_xlabel(viewingGrid.GetTable().keys[selectedCols[i]])
                currentAx.set_ylabel(viewingGrid.GetTable().keys[selectedCols[j]])                
                x += 1

        fig.tight_layout()
    
    def Hist2dPlotButtonClicked(self, event):
        '''Plot 2D histogram with linear fit (if p < 0.1).
        '''
        selectedCols = viewingGrid.GetSelectedCols()
        if len(selectedCols) < 2:
            return
        plotCount = sp.special.comb(len(selectedCols), 2)
        print(plotCount)
        if plotCount > 20:
            with wx.MessageDialog(self, "You are creating > 20 plots. Are you sure?", "Warning", style=wx.CENTER|wx.YES_NO|wx.NO_DEFAULT|wx.ICON_WARNING) as dlg:
                if dlg.ShowModal() == wx.ID_NO:
                    return
        
        rows = np.floor(np.sqrt(plotCount))
        cols = np.ceil(plotCount/rows)
        fig, axes = pyplot.subplots(nrows=int(rows), ncols=int(cols), squeeze=False)

        cmap = None
        x = 0
        for i in range(len(selectedCols)):
            for j in range(i + 1, len(selectedCols)):
                currentAx = axes.flat[x]
                currentAx.xaxis.get_major_formatter().set_powerlimits((-3, 3))
                currentAx.yaxis.get_major_formatter().set_powerlimits((-3, 3))

                counts, xedges, yedges, Image = currentAx.hist2d(viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[i]]],
                                 viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[j]]],
                                 bins=128,
                                 cmap=cmap)
                        
                pyplot.colorbar(mappable=Image, cmap=cmap, ax=currentAx, fraction=0.05)
                
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[i]]],viewingGrid.GetTable().cachedPipeline[viewingGrid.GetTable().keys[selectedCols[j]]])
#                print slope, intercept, r_value, p_value, std_err
                if p_value < 0.1:
                    fitX=[xedges[0], xedges[-1]]
                    fitY=[slope*xedges[0]+intercept, slope*xedges[-1]+intercept]
                    # currentAx.plot(fitX, fitY, linestyle='-', linewidth=3, color="w")
                    currentAx.plot(fitX, fitY, linestyle='--', linewidth=1, color="r", label="Linear Fit")
                    currentAx.annotate("r={:.3f}".format(r_value), (0.1, 0.9), xycoords="axes fraction", backgroundcolor='w', axes=currentAx)

                currentAx.set_xlabel(viewingGrid.GetTable().keys[selectedCols[i]])
                currentAx.set_ylabel(viewingGrid.GetTable().keys[selectedCols[j]])                
                x += 1
                
        fig.tight_layout()

    
    def AddMappingButtonClicked(self, event, key, mapping):
        '''Already doable from VisGUI. Just a nice GUI wrapper.
        '''
        key = key.GetValue().encode('ascii', 'ignore')
        mapping = mapping.GetValue().encode('ascii', 'ignore')

        if not hasattr(pipeline, "selectedDataSource") or not hasattr(pipeline.selectedDataSource, "setMapping"):
            with wx.MessageDialog(self, "Pipeline does not have a selectedDataSource, \nor the data source does not support setMapping.", "Error", style=wx.CENTER|wx.OK|wx.ICON_ERROR) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    return        
        pipeline.selectedDataSource.setMapping(key, mapping)
        # VisFr.UpdatePointColourChoices() # Broken, was to update points render in old verison of PYME
        self.RefreshButtonClicked(None)

class ViewingPanel(wx.Panel):
    '''Contains the grid only.
    '''
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        self.SetBackgroundColour(wx.BLUE)
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(ViewingGrid(self), proportion=1, flag=wx.EXPAND)
        self.SetSizer(mainSizer)

class ViewingGrid(wx.grid.Grid):
    '''Grid GUI. All formating and basic-click-responses handled automatically.
    '''
    def __init__(self, *args, **kwargs):
        wx.grid.Grid.__init__(self, *args, **kwargs)
        self.SetTable(PipelineGridBase(), True)
        self.EnableEditing(False)
        global viewingGrid
        viewingGrid = self
        self.AutoWidthByLabel()
        
        '''Change binding to left/right single/double click here.
        '''
        self.Bind(wx.grid.EVT_GRID_LABEL_RIGHT_CLICK, self.OnSort)
        self.Bind(wx.grid.EVT_GRID_CELL_RIGHT_CLICK, self.OnToggleCell)

    
    def Rebuild(self):
        '''Does not play well with changing data source. Have to do some manual cleanup.
        '''

        oldColCount = self.GetTable().colCount
        oldRowCount = self.GetTable().rowCount
        
        self.BeginBatch()
        self.Table.Build()
        
        for current, new, delmsg, addmsg in [
                (oldRowCount, self.GetTable().rowCount, wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED, wx.grid.GRIDTABLE_NOTIFY_ROWS_APPENDED),
                (oldColCount, self.GetTable().colCount, wx.grid.GRIDTABLE_NOTIFY_COLS_DELETED, wx.grid.GRIDTABLE_NOTIFY_COLS_APPENDED),
        ]:
            
            if new < current:
                msg = wx.grid.GridTableMessage(
                        self.GetTable(),
                        delmsg,
                        new,    # position
                        current-new,
                )                        
                self.ProcessTableMessage(msg)
            elif new > current:
                msg = wx.grid.GridTableMessage(
                        self.GetTable(),
                        addmsg,
                        new-current
                )
                self.ProcessTableMessage(msg)
                        
        self.AutoWidthByLabel()
        self.ForceRefresh()
        
        self.EndBatch()
    
    def AutoWidthByLabel(self):
        '''Cosmetic. Autowidth.
        '''
        self.SetDefaultColSize(self.GetDefaultColSize(), True)
        deviceContext = wx.ScreenDC()
        deviceContext.SetFont(self.GetLabelFont())
        for i in range(self.GetTable().colCount):
            width, height = deviceContext.GetMultiLineTextExtent(self.GetTable().keys[i])
            if width + 10 > self.GetDefaultColSize():
                self.SetColSize(i, width + 10)
    
    def OnSort(self, event):
        '''Pass on sort event to datasource.
        '''    
        if event.GetCol() >= 0:
            self.GetTable().Sort(event.GetCol())

        self.ForceRefresh()
    
    def OnToggleCell(self, event):
        '''Toggle highlighter.
        '''
        self.ClearSelection()
        self.SetGridCursor(event.GetRow(), event.GetCol())
        self.GetTable().ToggleHighlight(event.GetRow(), event.GetCol())
        self.ForceRefresh()

class PipelineGridBase(wx.grid.PyGridTableBase):
    '''Can't use standard Table object since slow to load all data to memory.
        Manually link to pipeline object. Handles sorting and highlight internally.
    '''
    def __init__(self, *args, **kwargs):        
        wx.grid.PyGridTableBase.__init__(self, *args, **kwargs)
        if pipeline == None:
            return
        self.highlighterPrecision = PipelineGridBase.GetDefaultHighlighterPrecision()
        self.Build()        
    
    def Build(self):
        #'live' pipeline is too slow
        
        try:
            # self.cachedPipeline = pipeline
            # self.cachedPipeline = dict(pipeline)
            self.cachedPipeline = pipeline.to_recarray()
        except:
            self.cachedPipeline = dict()
            for key in pipeline.keys():
                try:
                    self.cachedPipeline[key] = pipeline[key]
                except:
                    print("Key not found: " + key)

        self.keys = list(pipeline.keys())
        self.keys.sort()
        self.colCount = len(self.keys)
        # self.rowCount = max(len(v) for v in self.cachedPipeline.values())
        self.rowCount = len(self.cachedPipeline[self.keys[0]])
        self.sort = 0 #0 = unsort, 1 = asc, 2 = dsc
        self.sortCol = None
        self.sortIndex = np.arange(0, self.rowCount)

        self.attrDict = collections.OrderedDict()

        setupPanel.rowJumpTarget.SetRange(minVal=1, maxVal=self.rowCount)
        setupPanel.colJumpTarget.Clear()
        setupPanel.colJumpTarget.Append(self.keys)
#        print("rebuilt")
    
    def GetNumberRows(self):
        return self.rowCount
    
    def GetNumberCols(self):
        return self.colCount
        
    def IsEmptyCell(self, row, col):
        return False
        
    def GetValue(self, row, col):
        if col < len(self.keys) and row < len(self.cachedPipeline[self.keys[col]]):
            return self.cachedPipeline[self.keys[col]][self.sortIndex[row]]
        else:
            return "OOR"
        
    def SetValue(self, row, col, value):
        pass
    
    def GetAttr(self, row, col, kind):
        for k, v in reversed(self.attrDict):
            roundedValueKey = ("{:." + str(self.highlighterPrecision - 1) + "e}").format(self.GetValue(row, k))
            if roundedValueKey == v:
                index = 0 if k == col else 1
                self.attrDict[(k, roundedValueKey)][index].IncRef()
                return self.attrDict[(k, roundedValueKey)][index]                
        return wx.grid.GridCellAttr()
    
    def GetColLabelValue(self, col):
        returnStr = self.keys[col]
        if self.sortCol == col and not self.sort == 0:
            returnStr += "\n+" if self.sort == 1 else "\n-"
        return returnStr
        
    def Sort(self, col):
        if self.sortCol == col:
            if self.sort == 1:
                self.sortIndex = np.argsort(self.cachedPipeline[self.keys[self.sortCol]])[::-1]
                self.sort = 2
            elif self.sort == 2:
                self.sortCol = None
                self.sortIndex = np.arange(self.rowCount)
                self.sort = 0
        else:
            self.sortCol = col
            self.sortIndex = np.argsort(self.cachedPipeline[self.keys[self.sortCol]])
            self.sort = 1
        
    
    def ToggleHighlight(self, row, col):
        roundedValueKey = ("{:." + str(self.highlighterPrecision - 1) + "e}").format(self.GetValue(row, col))
        if (col, roundedValueKey) in self.attrDict:
            self.attrDict.pop((col, roundedValueKey))
        else:
            newAttrPrimary = wx.grid.GridCellAttr()
            newAttrSecondary = wx.grid.GridCellAttr()
            newAttrPrimary.SetBackgroundColour(wx.RED)
            np.random.seed(row + 1000 * col)
            rndColor = np.random.randint(low=128, high=255, size=3)
            newAttrPrimary.SetBackgroundColour(rndColor)
            rndColor -= 32
            newAttrSecondary.SetBackgroundColour(rndColor)
            self.attrDict[(col, roundedValueKey)] = (newAttrPrimary, newAttrSecondary)
    
    @staticmethod
    def GetDefaultHighlighterPrecision():
        return 3
    
    def SetHighlighterPrecision(self, value):
        self.attrDict.clear()
        self.highlighterPrecision = value
        self.View.ForceRefresh()


def Plug(visFr):
    '''Excute in VisGUI console to add this as a tab page.
    
    :param PYME.Analysis.LMVis.VisGUI.VisGUIFrame visFr: An instance of the main frame (```MainWindow```?).
        Used to add page and grab the pipeline.
    '''
    global VisFr
    global pipeline
    VisFr = visFr
    if hasattr(visFr, "pipeline"):
        pipeline = visFr.pipeline
    visFr.pipeViewerPanel = PipeViewerPanel(visFr)
    visFr.AddPage(visFr.pipeViewerPanel, caption="Pipeline Table Viewer")
