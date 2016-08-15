# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:35:18 2015

@author: david
"""

#from PYME.LMVis import pipelineView
import wx
import wx.html2

class ResultsView(object):
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.webview = wx.html2.WebView.New(dsviewer)

def Plug(dsviewer):
    dsviewer.sheetView = wx.html2.WebView.New(dsviewer)
    #dsviewer.updateHooks.append(dsviewer.view.Redraw)
    #dsviewer.sheetView = pipelineView.PipelinePanel(dsviewer, dsviewer.pipeline)
    #dsviewer.updateHooks.append(dsviewer.view.Redraw)
    dsviewer.AddPage(dsviewer.sheetView, True, 'Measurements')
    
    
