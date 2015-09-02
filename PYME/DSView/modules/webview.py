# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:26:33 2015

@author: david
"""

import wx
import wx.html2

def Plug(dsviewer):
    dsviewer.webview = wx.html2.WebView.New(dsviewer)
    #dsviewer.updateHooks.append(dsviewer.view.Redraw)
    dsviewer.AddPage(dsviewer.webview, True, 'Measurements')