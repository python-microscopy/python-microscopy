#!/usr/bin/python
##################
# displayPane.py
#
# Copyright David Baddeley, 2010
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
from collections import OrderedDict

import wx
import wx.lib.newevent

#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
# import pylab
from PYME.misc import colormaps
import numpy as np

from PYME.ui import histLimits

import PYME.config
from PYME.resources import getIconPath

DisplayInvalidEvent, EVT_DISPLAY_CHANGE = wx.lib.newevent.NewCommandEvent()

def CreateDisplayPane(panel, mapping, visFr):
    if PYME.config.get('VisGUI-new_layers', False):
        pane = DisplayPaneLight(panel, mapping, visFr)
    else:
        pane = DisplayPane(panel, mapping, visFr)
    panel.AddPane(pane)
    return pane

class DisplayPane(afp.foldingPane):
    def __init__(self, panel, glCanvas, visFr):
        afp.foldingPane.__init__(self, panel, -1, caption="Display", pinned = True)

        self.glCanvas = glCanvas
        self.visFr = visFr

        self.scaleBarLengths = OrderedDict([('<None>', None),
                                            ('50nm', 50),
                                            ('200nm', 200),
                                            ('500nm', 500),
                                            ('1um', 1000),
                                            ('5um', 5000)])
        self._pc_clim_change = False

        #Colourmap
        self._cmapnames = list(colormaps.cm.cmapnames)
        
        #print((cmapnames, self.glCanvas.cmap.name))

        #curCMapName = self.glCanvas.cmap.name
        curCMapName = 'gist_rainbow'

        #cmapReversed = False

        if curCMapName[-2:] == '_r':
            #cmapReversed = True
            curCMapName = curCMapName[:-2]

        cmInd = self._cmapnames.index(curCMapName)


        ##
        

        #box = wx.StaticBox(pan, -1, 'Colourmap:')
        #bsizer = wx.StaticBoxSizer(box)
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        
        #self.AddNewElement(self.r3DMode)
        
        pan = wx.Panel(self, -1)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # self.r3DMode = wx.RadioBox(pan, choices=['2D','3D'])
        # self.r3DMode.Bind(wx.EVT_RADIOBOX, self.OnChange3D)
        # hsizer.Add(self.r3DMode, 1, wx.ALL, 2)
        #
        # vsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(pan, wx.ID_ANY, 'LUT:'), 0, wx.ALL, 2)

        self.cColourmap = wx.Choice(pan, -1, choices=self._cmapnames)
        self.cColourmap.SetSelection(cmInd)

        hsizer.Add(self.cColourmap, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        #self.cbCmapReverse = wx.CheckBox(pan, -1, 'Invert')
        #self.cbCmapReverse.SetValue(cmapReversed)

        #hsizer.Add(self.cbCmapReverse, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #bsizer.Add(hsizer, 0, wx.ALL, 0)

        #bdsizer = wx.BoxSizer()
        #bdsizer.Add(bsizer, 1, wx.EXPAND|wx.ALL, 0)

        # cbLUTDraw = wx.CheckBox(pan, -1, 'Show')
        # cbLUTDraw.SetValue(self.glCanvas.LUTDraw)
        #
        # cbLUTDraw.Bind(wx.EVT_CHECKBOX, self.OnLUTDrawCB)
        #
        # hsizer.Add(cbLUTDraw, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        vsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        pan.SetSizerAndFit(vsizer)
        


        #self._pnl.AddFoldPanelWindow(self, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        self.AddNewElement(pan)

        self.cColourmap.Bind(wx.EVT_CHOICE, self.OnCMapChange)
        #self.cbCmapReverse.Bind(wx.EVT_CHECKBOX, self.OnCMapChange)


        #CLim
        pan = wx.Panel(self, -1)

        box = wx.StaticBox(pan, -1, 'Display range:')
        bsizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Min: '), 0, wx.LEFT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 1)

        self.tCLimMin = wx.TextCtrl(pan, -1, '%3.2f' % self.glCanvas.clim[0], size=(40,-1))
        hsizer.Add(self.tCLimMin, 1,wx.RIGHT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 1)

        hsizer.Add(wx.StaticText(pan, -1, '  Max: '), 0, wx.LEFT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 1)

        self.tCLimMax = wx.TextCtrl(pan, -1, '%3.2f' % self.glCanvas.clim[1], size=(40,-1))
        hsizer.Add(self.tCLimMax, 1, wx.RIGHT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL,1)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        self.hlCLim = histLimits.HistLimitPanel(pan, -1, np.array(self.glCanvas.c), self.glCanvas.clim[0], self.glCanvas.clim[1], size=(150, 100))
        bsizer.Add(self.hlCLim, 0, wx.ALL|wx.EXPAND, 1)


#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#        self.tPercentileCLim = wx.TextCtrl(pan, -1, '.95', size=(40,-1))
#        hsizer.Add(self.tPercentileCLim, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        bPercentile = wx.Button(pan, -1, 'Set Percentile')
#        hsizer.Add(bPercentile, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#
#        bsizer.Add(hsizer, 0, wx.ALL, 0)

        bdsizer = wx.BoxSizer()
        bdsizer.Add(bsizer, 1, wx.EXPAND|wx.ALL, 0)

        pan.SetSizer(bdsizer)
        bdsizer.Fit(pan)

        #self.hlCLim.Refresh()

        self.AddNewElement(pan)
        #self._pnl.AddFoldPanelWindow(self, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.tCLimMin.Bind(wx.EVT_TEXT, self.OnCLimChange)
        self.tCLimMax.Bind(wx.EVT_TEXT, self.OnCLimChange)

        self.hlCLim.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnCLimHistChange)

        #bPercentile.Bind(wx.EVT_BUTTON, self.OnPercentileCLim)

        #self._pnl.AddFoldPanelSeparator(self)


        #LUT
        


        #Scale Bar
        # pan = wx.Panel(self, -1)
        #
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #
        # hsizer.Add(wx.StaticText(pan, -1, 'Scale Bar: '), 0, wx.LEFT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)
        #
        #
        # chInd = list(self.scaleBarLengths.values()).index(self.glCanvas.scaleBarLength)
        #
        # chScaleBar = wx.Choice(pan, -1, choices = list(self.scaleBarLengths.keys()))
        # chScaleBar.SetSelection(chInd)
        # hsizer.Add(chScaleBar, 0,wx.RIGHT|wx.TOP|wx.BOTTOM|wx.ALIGN_CENTER_VERTICAL, 5)
        #
        # pan.SetSizer(hsizer)
        # hsizer.Fit(pan)
        #
        # self.AddNewElement(pan)
        # #self._pnl.AddFoldPanelWindow(self, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        #
        # chScaleBar.Bind(wx.EVT_CHOICE, self.OnChangeScaleBar)

        #self._pnl.AddPane(self)


    def OnCMapChange(self, event):
        cmapname = self._cmapnames[self.cColourmap.GetSelection()]
        #if self.cbCmapReverse.GetValue():
        #    cmapname += '_r'

        self.glCanvas.setCMap(colormaps.cm[cmapname])
        #self.visFr.OnGLViewChanged()
        evt = DisplayInvalidEvent(self.GetId())
        self.ProcessEvent(evt)

    def OnLUTDrawCB(self, event):
        self.glCanvas.LUTDraw = event.IsChecked()
        self.glCanvas.Refresh()

    def OnChangeScaleBar(self, event):
        self.glCanvas.scaleBarLength = self.scaleBarLengths[event.GetString()]
        self.glCanvas.Refresh()
        
    def OnChange3D(self, event):
        self.glCanvas.displayMode = self.r3DMode.GetString(self.r3DMode.GetSelection())
        self.glCanvas.Refresh()

    def OnCLimChange(self, event):
        if self._pc_clim_change: #avoid setting CLim twice
            self._pc_clim_change = False #clear flag
        else:
            cmin = float(self.tCLimMin.GetValue())
            cmax = float(self.tCLimMax.GetValue())

            self.hlCLim.SetValue((cmin, cmax))

            self.glCanvas.setCLim((cmin, cmax))
            
            evt = DisplayInvalidEvent(self.GetId())
            self.ProcessEvent(evt)

    def OnCLimHistChange(self, event):
        self.glCanvas.setCLim((event.lower, event.upper))
        self._pc_clim_change = True
        self.tCLimMax.SetValue('%3.2f' % self.glCanvas.clim[1])
        self._pc_clim_change = True
        self.tCLimMin.SetValue('%3.2f' % self.glCanvas.clim[0])
        evt = DisplayInvalidEvent(self.GetId())
        self.ProcessEvent(evt)

    def OnPercentileCLim(self, event):
        self.hlCLim.SetMinMax()
        #pc = .95#float(self.tPercentileCLim.GetValue())

        #self.glCanvas.setPercentileCLim(pc)

        #self._pc_clim_change = True
        #self.tCLimMax.SetValue('%3.2f' % self.glCanvas.clim[1])
        #self._pc_clim_change = True
        #self.tCLimMin.SetValue('%3.2f' % self.glCanvas.clim[0])

        #self.hlCLim.SetValue(self.glCanvas.clim)


class DisplayPaneLight(afp.foldingPane):
    def __init__(self, panel, glCanvas, visFr):
        afp.foldingPane.__init__(self, panel, -1, caption="Display", pinned=True)
        
        self.glCanvas = glCanvas
        self.visFr = visFr
        
        self.scaleBarLengths = OrderedDict([('<None>', None),
                                            ('50nm', 50),
                                            ('200nm', 200),
                                            ('500nm', 500),
                                            ('1um', 1000),
                                            ('5um', 5000)])
        
        pan = wx.Panel(self, -1)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.r3DMode = wx.RadioBox(pan, choices=['2D', '3D'])
        self.r3DMode.Bind(wx.EVT_RADIOBOX, self.OnChange3D)
        hsizer.Add(self.r3DMode, 1, wx.ALL, 2)
        
        vsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(pan, wx.ID_ANY, 'LUT:'), 0, wx.ALL, 2)
        
        cbLUTDraw = wx.CheckBox(pan, -1, 'Show')
        cbLUTDraw.SetValue(self.glCanvas.LUTDraw)
        
        cbLUTDraw.Bind(wx.EVT_CHECKBOX, self.OnLUTDrawCB)
        
        hsizer.Add(cbLUTDraw, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        
        vsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 0)
        
        pan.SetSizerAndFit(vsizer)
        
        #self._pnl.AddFoldPanelWindow(self, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        self.AddNewElement(pan)
        
        #Scale Bar
        pan = wx.Panel(self, -1)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(pan, -1, 'Scale Bar: '), 0, wx.LEFT | wx.TOP | wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, 5)
        
        chInd = list(self.scaleBarLengths.values()).index(self.glCanvas.scaleBarLength)
        
        chScaleBar = wx.Choice(pan, -1, choices=list(self.scaleBarLengths.keys()))
        chScaleBar.SetSelection(chInd)
        hsizer.Add(chScaleBar, 0, wx.RIGHT | wx.TOP | wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, 5)
        
        pan.SetSizer(hsizer)
        hsizer.Fit(pan)
        
        self.AddNewElement(pan)
        #self._pnl.AddFoldPanelWindow(self, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        
        chScaleBar.Bind(wx.EVT_CHOICE, self.OnChangeScaleBar)
        
    
    def OnLUTDrawCB(self, event):
        self.glCanvas.LUTDraw = event.IsChecked()
        self.glCanvas.Refresh()
    
    def OnChangeScaleBar(self, event):
        self.glCanvas.scaleBarLength = self.scaleBarLengths[event.GetString()]
        self.glCanvas.Refresh()
    
    def OnChange3D(self, event):
        self.glCanvas.displayMode = self.r3DMode.GetString(self.r3DMode.GetSelection())
        self.glCanvas.Refresh()


class DisplayPaneHorizontal(wx.Panel):
    def __init__(self, parent, glCanvas, visFr):
        wx.Panel.__init__(self, parent, -1)
        
        self.glCanvas = glCanvas
        self.visFr = visFr
        
        self.scaleBarLengths = OrderedDict([('<None>', None),
                                            ('50nm', 50),
                                            ('200nm', 200),
                                            ('500nm', 500),
                                            ('1um', 1000),
                                            ('5um', 5000)])
        
        
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.r3DMode = wx.RadioBox(self, choices=['2D', '3D'])
        self.r3DMode.Bind(wx.EVT_RADIOBOX, self.OnChange3D)
        hsizer.Add(self.r3DMode, 0, wx.LEFT|wx.RIGHT| wx.ALIGN_CENTER_VERTICAL, 2)
        
        hsizer.AddSpacer(10)
        
        bTop = wx.BitmapButton(self, -1, wx.Bitmap(getIconPath('view-top.png')), style=wx.NO_BORDER|wx.BU_AUTODRAW, name='Top')
        bTop.Bind(wx.EVT_BUTTON, self.OnViewTop)
        hsizer.Add(bTop, 0, wx.LEFT|wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 0)
        bFront = wx.BitmapButton(self, -1, wx.Bitmap(getIconPath('view-front.png')), style=wx.NO_BORDER|wx.BU_AUTODRAW, name='Front')
        bFront.Bind(wx.EVT_BUTTON, self.OnViewFront)
        hsizer.Add(bFront, 0, wx.LEFT|wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 0)
        bRight = wx.BitmapButton(self, -1, wx.Bitmap(getIconPath('view-right.png')), style=wx.NO_BORDER|wx.BU_AUTODRAW, name='Right')
        bRight.Bind(wx.EVT_BUTTON, self.OnViewRight)
        hsizer.Add(bRight, 0, wx.LEFT|wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 0)
        bHome = wx.BitmapButton(self, -1, wx.Bitmap(getIconPath('view-home.png')), style=wx.NO_BORDER | wx.BU_AUTODRAW, name='Home')
        bHome.Bind(wx.EVT_BUTTON, self.visFr.OnHome)
        hsizer.Add(bHome, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 0)
        
        hsizer.AddSpacer(10)
        
        cbLUTDraw = wx.ToggleButton(self, -1, 'LUT', style=wx.BU_EXACTFIT)
        #cbLUTDraw.SetBitmap(wx.Bitmap(getIconPath('LUT.png')))
        #cbLUTDraw = wx.BitmapToggleButton(self, -1, wx.Bitmap(getIconPath('LUT.png')))
        cbLUTDraw.SetValue(self.glCanvas.LUTDraw)
        
        #cbLUTDraw.Bind(wx.EVT_CHECKBOX, self.OnLUTDrawCB)
        cbLUTDraw.Bind(wx.EVT_TOGGLEBUTTON, self.OnLUTDrawCB)
        
        hsizer.Add(cbLUTDraw, 0, wx.LEFT|wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)

        hsizer.AddSpacer(10)
        
        #Scale Bar
        
        hsizer.Add(wx.StaticText(self, -1, 'Scale Bar: '), 0, wx.LEFT  | wx.ALIGN_CENTER_VERTICAL, 5)
        
        chInd = list(self.scaleBarLengths.values()).index(self.glCanvas.scaleBarLength)
        
        chScaleBar = wx.Choice(self, -1, choices=list(self.scaleBarLengths.keys()))
        chScaleBar.SetSelection(chInd)
        hsizer.Add(chScaleBar, 0, wx.RIGHT  | wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.AddSpacer(10)

        #Background colour

        hsizer.Add(wx.StaticText(self, -1, 'BG Colour: '), 0, wx.LEFT  | wx.ALIGN_CENTER_VERTICAL,5)

        colour_ctrl = wx.ColourPickerCtrl(self)
        colour_ctrl.Bind(wx.EVT_COLOURPICKER_CHANGED, self.OnColourChanged)
        hsizer.Add(colour_ctrl, 0, wx.RIGHT  | wx.ALIGN_CENTER_VERTICAL, 5)
        
        self.SetSizerAndFit(hsizer)
        
        #self._pnl.AddFoldPanelWindow(self, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        
        chScaleBar.Bind(wx.EVT_CHOICE, self.OnChangeScaleBar)
        
    def OnColourChanged(self, event):
        self.glCanvas.clear_colour = np.array(event.GetColour().Get(True))/255.
        self.glCanvas.Refresh()
    
    def OnLUTDrawCB(self, event):
        self.glCanvas.LUTDraw = event.IsChecked()
        self.glCanvas.Refresh()
    
    def OnChangeScaleBar(self, event):
        self.glCanvas.scaleBarLength = self.scaleBarLengths[event.GetString()]
        self.glCanvas.Refresh()
    
    def OnChange3D(self, event):
        self.glCanvas.displayMode = self.r3DMode.GetString(self.r3DMode.GetSelection())
        self.glCanvas.Refresh()
        
    def OnViewTop(self, event):
        self.glCanvas.view.top()
        self.glCanvas.Refresh()
        
    def OnViewFront(self, event):
        self.glCanvas.view.front()
        self.glCanvas.Refresh()
        
    def OnViewRight(self, event):
        self.glCanvas.view.right()
        self.glCanvas.Refresh()
