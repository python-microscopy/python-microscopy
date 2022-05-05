#!/usr/bin/python

##################
# dSimControl.py
#
# Copyright David Baddeley, 2009
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

#Boa:Dialog:dSimControl

import json
import logging

import numpy as np
import scipy
import wx
import wx.grid

import PYME.ui.manualFoldPanel as afp
from PYME.simulation import wormlike2
from . import fluor
from . import rend_im

logger = logging.getLogger(__name__)

def create(parent):
    return dSimControl(parent)

[wxID_DSIMCONTROL, wxID_DSIMCONTROLBGENFLOURS, wxID_DSIMCONTROLBGENWORMLIKE, 
 wxID_DSIMCONTROLBLOADPOINTS, wxID_DSIMCONTROLBPAUSE, 
 wxID_DSIMCONTROLBSAVEPOINTS, wxID_DSIMCONTROLCBCOLOUR, 
 wxID_DSIMCONTROLCBFLATTEN, wxID_DSIMCONTROLGPROBE, wxID_DSIMCONTROLGSPONTAN, 
 wxID_DSIMCONTROLGSWITCH, wxID_DSIMCONTROLNTRANSITIONTENSOR, 
 wxID_DSIMCONTROLSTATICBOX1, wxID_DSIMCONTROLSTATICBOX2, 
 wxID_DSIMCONTROLSTATICBOX3, wxID_DSIMCONTROLSTATICBOX4, 
 wxID_DSIMCONTROLSTATICBOX5, wxID_DSIMCONTROLSTATICTEXT1, 
 wxID_DSIMCONTROLSTATICTEXT2, wxID_DSIMCONTROLSTATICTEXT3, 
 wxID_DSIMCONTROLSTATICTEXT4, wxID_DSIMCONTROLSTATICTEXT5, 
 wxID_DSIMCONTROLSTCUROBJPOINTS, wxID_DSIMCONTROLSTSTATUS, 
 wxID_DSIMCONTROLTEXPROBE, wxID_DSIMCONTROLTEXSWITCH, wxID_DSIMCONTROLTKBP, 
 wxID_DSIMCONTROLTNUMFLUOROPHORES, 
] = [wx.NewId() for _init_ctrls in range(28)]

[wxID_DSIMCONTROLTREFRESH] = [wx.NewId() for _init_utils in range(1)]


from PYME.recipes.traits import HasTraits, Float, Dict, Bool, List
class PSFSettings(HasTraits):
    wavelength_nm = Float(700.)
    NA = Float(1.47)
    vectorial = Bool(False)
    zernike_modes = Dict()
    zernike_modes_lower = Dict()
    phases = List([0, .5, 1, 1.5])
    four_pi = Bool(False)
    
    def default_traits_view(self):
        from traitsui.api import View, Item
        #from PYME.ui.custom_traits_editors import CBEditor
    
        return View(Item(name='wavelength_nm'),
                    Item(name='NA'),
                    Item(name='vectorial'),
                    Item(name='four_pi', label='4Pi'),
                    Item(name='zernike_modes'),
                    Item(name='zernike_modes_lower', visible_when='four_pi==True'),
                    Item(name='phases', visible_when='four_pi==True', label='phases/pi'),
                    resizable = True,
                    buttons   = [ 'OK' ])
        
        

class dSimControl(afp.foldPanel):
    def _init_coll_nTransitionTensor_Pages(self, parent):
        # generated method, don't edit

        parent.AddPage(imageId=-1, page=self.gSpontan, select=True,
              text='Spontaneous')
        parent.AddPage(imageId=-1, page=self.gSwitch, select=False,
              text='Switching Laser')
        parent.AddPage(imageId=-1, page=self.gProbe, select=False,
              text='Probe Laser')

    def _init_utils(self):
        #pass
        # generated method, don't edit
        self.tRefresh = wx.Timer(id=wxID_DSIMCONTROLTREFRESH, owner=self)
        self.Bind(wx.EVT_TIMER, self.OnTRefreshTimer,
              id=wxID_DSIMCONTROLTREFRESH)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        #wx.Panel.__init__(self, id=-1, parent=prnt)#, size=wx.Size(442, 637))
        self._init_utils()
        #self.SetClientSize(wx.Size(434, 610))
        #vsizer= wx.BoxSizer(wx.VERTICAL)

        ################ Splitter ######################

        item = afp.foldingPane(self, -1, caption="Virtual Hardware", pinned=True)

        #sbsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Image splitting & PSF'),
        #                            wx.VERTICAL)
        
        pane = wx.Panel(item, -1)
        
        sbsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pane, -1, 'Number of detection channels: '), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        self.cNumSplitterChans = wx.Choice(pane, -1, choices=['1 - Standard', '2 - Ratiometric/Biplane', '4 - HT / 4Pi-SMS'])
        self.cNumSplitterChans.SetSelection(0)
        self.cNumSplitterChans.Bind(wx.EVT_CHOICE, self.OnNumChannelsChanged)
        hsizer.Add(self.cNumSplitterChans, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        sbsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        self.gSplitter = wx.grid.Grid(pane, -1)
        self.setupSplitterGrid()
        self.OnNumChannelsChanged()
        sbsizer.Add(self.gSplitter, 0, wx.RIGHT | wx.EXPAND, 2)
        
        sbsizer.AddSpacer(8)

        ############## PSF Settings ################
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.st_psf = wx.StaticText(pane, -1, 'PSF: Default widefield')
        hsizer.Add(self.st_psf, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        
        hsizer.AddStretchSpacer()

        self.bSetTPSF = wx.Button(pane, -1, 'Set Theoretical')
        self.bSetTPSF.Bind(wx.EVT_BUTTON, self.OnBSetPSFModel)
        hsizer.Add(self.bSetTPSF, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)

        self.bSetPSF = wx.Button(pane, -1, 'Set Experimental')
        self.bSetPSF.Bind(wx.EVT_BUTTON, self.OnBSetPSF)
        hsizer.Add(self.bSetPSF, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)

        self.bSavePSF = wx.Button(pane, -1, 'Save')
        self.bSavePSF.Bind(wx.EVT_BUTTON, self.OnBSavePSF)
        hsizer.Add(self.bSavePSF, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)

        sbsizer.Add(hsizer, 0, wx.ALL, 2)

        sbsizer.AddSpacer(8)

        pane.SetSizerAndFit(sbsizer)

        item.AddNewElement(pane)
        self.AddPane(item)

        #vsizer.Add(sbsizer, 0, wx.ALL | wx.EXPAND, 2)


        ########### Fluorophore Positions ############        
        #sbsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Fluorophore Postions'),
        #                          wx.VERTICAL)

        item = afp.foldingPane(self, -1, caption='Fluorophore Postions', pinned=True)
        pane = wx.Panel(item, -1)
        sbsizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer=wx.BoxSizer(wx.HORIZONTAL)

        self.tNumFluorophores = wx.TextCtrl(pane, -1, value='10000', size=(60, -1))
        hsizer.Add(self.tNumFluorophores, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        hsizer.Add(wx.StaticText(pane,-1,'fluorophores distributed evenly along'),
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.tKbp = wx.TextCtrl(pane, -1, size=(60, -1), value='200000')
        hsizer.Add(self.tKbp, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        hsizer.Add(wx.StaticText(pane,-1,'nm'),
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
                   
        hsizer.AddStretchSpacer()

        self.bGenWormlike = wx.Button(pane, -1,'Generate')
        self.bGenWormlike.Bind(wx.EVT_BUTTON, self.OnBGenWormlikeButton)
        hsizer.Add(self.bGenWormlike, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 2)
        hsizer=wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(pane,-1,'Persistence length [nm]:'),
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        self.tPersist = wx.TextCtrl(pane, -1, size=(60, -1), value='1500')
        hsizer.Add(self.tPersist, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        hsizer.Add(wx.StaticText(pane,-1,'Z scale:'),
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        self.tZScale = wx.TextCtrl(pane, -1, size=(60, -1), value='1.0')
        hsizer.Add(self.tZScale, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.cbFlatten = wx.CheckBox(pane, -1, 'flatten (set z to 0)')
        self.cbFlatten.SetValue(False)
        hsizer.Add(self.cbFlatten, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        sbsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.cbColour = wx.CheckBox(pane, -1, u'Colourful')
        self.cbColour.SetValue(False)
        hsizer.Add(self.cbColour, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        self.cbWrap = wx.CheckBox(pane, -1, u'Wrap at FOV edge')
        self.cbWrap.SetValue(True)
        hsizer.Add(self.cbWrap, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        hsizer=wx.BoxSizer(wx.HORIZONTAL)
        
        self.stCurObjPoints = wx.StaticText(pane, -1, 'Current object has NO points')
        self.stCurObjPoints.SetForegroundColour(wx.RED)
        hsizer.Add(self.stCurObjPoints, 0, wx.ALL, 2)
        hsizer.AddStretchSpacer()

        self.bLoadPoints = wx.Button(pane, -1,'Load From File')
        self.bLoadPoints.Bind(wx.EVT_BUTTON, self.OnBLoadPointsButton)
        hsizer.Add(self.bLoadPoints, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.bSavePoints = wx.Button(pane, -1,'Save To File')
        self.bSavePoints.Bind(wx.EVT_BUTTON, self.OnBSavePointsButton)
        hsizer.Add(self.bSavePoints, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 2)

        pane.SetSizerAndFit(sbsizer)
        item.AddNewElement(pane)
        self.AddPane(item)
        #vsizer.Add(sbsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        

        
        ################ Virtual Fluorophores ###########
        
        #sbsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Generate Virtual Fluorophores'),
        #                          wx.VERTICAL)
        
        item = afp.foldingPane(self, -1, caption='Fluorophore switching model', pinned=True)
        pane = wx.Panel(item, -1)
        sbsizer = wx.BoxSizer(wx.VERTICAL)

        self.nSimulationType = wx.Notebook(pane, -1)

        ######################## Based on first principles... #########
        pFirstPrinciples = wx.Panel(self.nSimulationType, -1)
        pFirstPrinciplesSizer = wx.BoxSizer(wx.VERTICAL)

        sbsizer2 = wx.StaticBoxSizer(
            wx.StaticBox(pFirstPrinciples, -1, 'Transition Tensor'),
            wx.VERTICAL)

        self.nTransitionTensor = wx.Notebook(pFirstPrinciples, -1)
        # self.nTransitionTensor.SetLabel('Transition Probabilites')

        #pFirstPrinciplesSizer.Add(wx.StaticText(pFirstPrinciples, -1, "A 4-state 1st order kinetic model, this allows simulation of all common modalities (PALM, STORM, PAINT, etc ...) with suitable parameter choices"), 0, wx.ALL, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(pFirstPrinciples, -1, 'Generate matrix for:'), 0, wx.ALL|wx.ALIGN_CENTRE_HORIZONTAL, 2)
        
        self.cModelPresets = wx.Choice(pFirstPrinciples, -1, choices=['STORM', 'PALM', 'PAINT'])
        self.cModelPresets.Bind(wx.EVT_CHOICE, self.OnModelPresets)

        hsizer.Add(self.cModelPresets, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        
        sbsizer2.Add(hsizer, 0, wx.ALL, 2)
        
        self.gSpontan = wx.grid.Grid(self.nTransitionTensor, -1)
        self.gSwitch = wx.grid.Grid(self.nTransitionTensor, -1)
        self.gProbe = wx.grid.Grid(self.nTransitionTensor, -1)

        self._init_coll_nTransitionTensor_Pages(self.nTransitionTensor)

        self.setupGrid(self.gSpontan, self.states, self.stateTypes)
        self.setupGrid(self.gSwitch, self.states, self.stateTypes)
        self.setupGrid(self.gProbe, self.states, self.stateTypes)

        sbsizer2.Add(self.nTransitionTensor, 1, wx.EXPAND | wx.ALL, 2)
        pFirstPrinciplesSizer.Add(sbsizer2, 1, wx.EXPAND | wx.ALL, 2)

        sbsizer2 = wx.StaticBoxSizer(
            wx.StaticBox(pFirstPrinciples, -1, 'Excitation Crossections (Fluorophore Brightness)'),
            wx.HORIZONTAL)

        sbsizer2.Add(wx.StaticText(pFirstPrinciples, -1, 'Switching Laser:'), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)

        self.tExSwitch = wx.TextCtrl(pFirstPrinciples, -1, value='1')
        sbsizer2.Add(self.tExSwitch, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)

        sbsizer2.Add(
            wx.StaticText(pFirstPrinciples, -1, 'photons/mWs     Probe Laser:'), 0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)

        self.tExProbe = wx.TextCtrl(pFirstPrinciples, -1, value='100')
        sbsizer2.Add(self.tExProbe, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)

        sbsizer2.Add(wx.StaticText(pFirstPrinciples, -1, 'photons/mWs'), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        self.sbsizer2 = sbsizer2

        pFirstPrinciplesSizer.Add(sbsizer2, 0, wx.EXPAND | wx.ALL, 2)

        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bGenFlours = wx.Button(pFirstPrinciples, -1, 'Go')
        self.bGenFlours.Bind(wx.EVT_BUTTON, self.OnBGenFloursButton)
        hsizer.Add(self.bGenFlours, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)

        pFirstPrinciplesSizer.Add(hsizer, 0, wx.ALL | wx.ALIGN_RIGHT, 2)
        pFirstPrinciples.SetSizer(pFirstPrinciplesSizer)

        ######################## Based on empirical data... #########

        pEmpiricalModel = wx.Panel(self.nSimulationType, -1)
        pEmpiricalModelSizer = wx.BoxSizer(wx.VERTICAL)

        sbsizer2 = wx.StaticBoxSizer(wx.StaticBox(pEmpiricalModel, -1,
                                                  'Load Dye Kinetics Histogram (JSON)'),
                                     wx.HORIZONTAL)

        self.stEmpiricalHist = wx.StaticText(pEmpiricalModel, -1, 'File: ')
        sbsizer2.Add(self.stEmpiricalHist, 0, wx.ALL, 2)
        sbsizer2.AddStretchSpacer()

        self.bLoadEmpiricalHist = wx.Button(pEmpiricalModel, -1, 'Load')
        self.bLoadEmpiricalHist.Bind(wx.EVT_BUTTON, self.OnBLoadEmpiricalHistButton)
        sbsizer2.Add(self.bLoadEmpiricalHist, 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT, 2)

        pEmpiricalModelSizer.Add(sbsizer2, 0, wx.ALL | wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bGenEmpiricalHistFluors = wx.Button(pEmpiricalModel, -1, 'Go')
        self.bGenEmpiricalHistFluors.Bind(wx.EVT_BUTTON, self.OnBGenEmpiricalHistFluorsButton)
        hsizer.Add(self.bGenEmpiricalHistFluors, 1,
                   wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT, 2)

        pEmpiricalModelSizer.Add(hsizer, 0, wx.ALL | wx.ALIGN_RIGHT, 2)

        pEmpiricalModel.SetSizer(pEmpiricalModelSizer)

        
        self.pFirstPrinciples = pFirstPrinciples
        self.pEmpericalModel = pEmpiricalModel
        
        self.nSimulationType.AddPage(imageId=-1, page=pFirstPrinciples,
                                     select=True,
                                     text='Theoretical State Model')
        self.nSimulationType.AddPage(imageId=-1, page=pEmpiricalModel,
                                     select=False,
                                     text='Data Based Empirical Model')
        sbsizer.Add(self.nSimulationType, 0, wx.ALL | wx.EXPAND, 2)

        #vsizer.Add(sbsizer, 0, wx.ALL | wx.EXPAND, 2)
        pane.SetSizerAndFit(sbsizer)
        item.AddNewElement(pane)
        self.AddPane(item)

        ######## Status #########
        
        #sbsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Status'),
        #                          wx.VERTICAL)

        item = afp.foldingPane(self, -1, caption='Status', pinned=True)
        pane = wx.Panel(item, -1)
        sbsizer = wx.BoxSizer(wx.VERTICAL)

        self.stStatus = wx.StaticText(pane, -1,
              label='hello\nworld\n\n\nfoo')
        sbsizer.Add(self.stStatus, 0, wx.ALL|wx.EXPAND, 2)

        self.bPause = wx.Button(pane, -1,'Pause')
        self.bPause.Bind(wx.EVT_BUTTON, self.OnBPauseButton)
        sbsizer.Add(self.bPause, 0, wx.ALL|wx.ALIGN_RIGHT, 2)

        pane.SetSizerAndFit(sbsizer)
        item.AddNewElement(pane)
        self.AddPane(item)
        
        #vsizer.Add(sbsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        #self.vsizer = vsizer



        

    def setupGrid(self, grid, states, stateTypes):
        nStates = len(states)
        
        grid.SetDefaultColSize(70)
        grid.CreateGrid(nStates, nStates)
        
        for i in range(nStates):
            grid.SetRowLabelValue(i, states[i])
            grid.SetColLabelValue(i, states[i])
            grid.SetReadOnly(i, i)
            grid.SetCellBackgroundColour(i, i, wx.LIGHT_GREY)
            grid.SetCellTextColour(i, i, wx.LIGHT_GREY)
            
            if (stateTypes[i] == fluor.TO_ONLY):
                for j in range(nStates):
                    grid.SetReadOnly(i, j)
                    grid.SetCellBackgroundColour(i, j, wx.LIGHT_GREY)
                    grid.SetCellTextColour(i, j, wx.LIGHT_GREY)
            
            if (stateTypes[i] == fluor.FROM_ONLY):
                for j in range(nStates):
                    grid.SetReadOnly(j, i)
                    grid.SetCellBackgroundColour(j, i, wx.LIGHT_GREY)
                    grid.SetCellTextColour(j, i, wx.LIGHT_GREY)
                    
        grid.SetMinSize(grid.BestSize)
        #print grid.BestSize, grid.MinSize
        #grid.GetParent().GetParent().Layout()
                    
    
    def fillGrids(self, vals):
        nStates = len(self.states)
        for i in range(nStates):
            for j in range(nStates):
                self.gSpontan.SetCellValue(i,j, '%f' % vals[i,j,0]) 
                self.gSwitch.SetCellValue(i,j, '%f' % vals[i,j,1])
                self.gProbe.SetCellValue(i,j, '%f' % vals[i,j,2])
                
    def getTensorFromGrids(self):
        nStates = len(self.states)
        transTens = scipy.zeros((nStates,nStates,3))
        
        for i in range(nStates):
            for j in range(nStates):
                transTens[i,j,0] = float(self.gSpontan.GetCellValue(i,j))
                transTens[i,j,1] = float(self.gSwitch.GetCellValue(i,j))
                transTens[i,j,2] = float(self.gProbe.GetCellValue(i,j))
        
        return transTens

    def setupSplitterGrid(self):
        grid = self.gSplitter

        #grid.SetDefaultColSize(70)
        grid.CreateGrid(2, 4)

        grid.SetRowLabelValue(0, 'Spectral chan')
        grid.SetRowLabelValue(1, 'Z offset [nm]')

        #hard code some initial values
        spec = [0, 1, 1, 0]
        z_offset = [0, -200, 300., 500.]

        for i in range(4):
            grid.SetColLabelValue(i, 'Chan %d' % i)

            grid.SetCellValue(0, i, '%d' % spec[i])
            grid.SetCellValue(1, i, '%3.2f' % z_offset[i])

            #grid.SetCellTextColour(i, i, wx.LIGHT_GREY)

    def getSplitterInfo(self):
        nChans = int(self.cNumSplitterChans.GetStringSelection()[0])

        zOffsets = [float(self.gSplitter.GetCellValue(1, i)) for i in range(nChans)]
        specChans = [int(self.gSplitter.GetCellValue(0, i)) for i in range(nChans)]

        return zOffsets, specChans

        
    
    def __init__(self, parent, scope=None, states=['Caged', 'On', 'Blinked', 'Bleached'], stateTypes=[fluor.FROM_ONLY, fluor.ALL_TRANS, fluor.ALL_TRANS, fluor.TO_ONLY], startVals=None, activeState=fluor.states.active):
        afp.foldPanel.__init__(self, parent, -1)
        #self._init_ctrls(parent)
        
        self.states = states
        self.stateTypes = stateTypes
        self.activeState = activeState
        self.scope = scope

        self._init_ctrls(parent)
        
        # self.setupGrid(self.gSpontan, states, stateTypes)
        # self.setupGrid(self.gSwitch, states, stateTypes)
        # self.setupGrid(self.gProbe, states, stateTypes)
        
        if (startVals is None): #use defaults
            startVals = fluor.createSimpleTransitionMatrix()
            
        self.fillGrids(startVals)
        
        self.spectralSignatures = scipy.array([[1, 0.3], [.7, .7], [0.2, 1]])

        
        self.points = []
        self.EmpiricalHist = None
        self.tRefresh.Start(200)

        #self.OnBGenWormlikeButton(None)
        #self.SetSizerAndFit(self.vsizer)
        

    def OnBGenWormlikeButton(self, event):
        import numpy as np
        kbp = float(self.tKbp.GetValue())
        numFluors = int(self.tNumFluorophores.GetValue())
        persistLength= float(self.tPersist.GetValue())
        #wc = wormlike2.fibre30nm(kbp, 10*kbp/numFluors)
        wc = wormlike2.wiglyFibre(kbp, persistLength, kbp / numFluors)

        XVals = self.scope.cam.XVals
        YVals = self.scope.cam.YVals

        x_pixels = len(XVals)
        y_pixels = len(YVals)

        #numChans = 1 + int(self.cbColour.GetValue())

        numChans = int(self.cNumSplitterChans.GetStringSelection()[0])

        x_chan_pixels = int(x_pixels/numChans)
        x_chan_size = XVals[x_chan_pixels-1] - XVals[0]

        y_chan_size = YVals[-1] - YVals[0]

        wc.xp = wc.xp - wc.xp.mean() + x_chan_size/2
        wc.yp = wc.yp - wc.yp.mean() + y_chan_size/2

        if self.cbWrap.GetValue():
            wc.xp = np.mod(wc.xp, x_chan_size) + XVals[0]
            wc.yp = np.mod(wc.yp, y_chan_size) + YVals[0]

        if self.cbFlatten.GetValue():
            wc.zp *= 0
        else:
            wc.zp -= wc.zp.mean()
            wc.zp *= float(self.tZScale.GetValue())
        
        self.points = []
        
        for i in range(len(wc.xp)):
            if self.cbColour.GetValue():
                self.points.append((wc.xp[i],wc.yp[i],wc.zp[i], float(i/((len(wc.xp) + 1)/3))))
            else:
                self.points.append((wc.xp[i],wc.yp[i],wc.zp[i], 0))

        
        self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        self.stCurObjPoints.SetForegroundColour("dark green")
        self.scope.cam.setFluors(None)

        self._generate_and_set_fluorophores()
        #event.Skip()

    def OnBLoadPointsButton(self, event):
        fn = wx.FileSelector('Read point positions from file')
        if fn is None:
            logger.warning('No file selected, cancelling')
            return

        if fn.endswith('.npy'):
            self.points = list(np.load(fn))
        else:
            self.points = np.loadtxt(fn)

        self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        self.stCurObjPoints.SetForegroundColour("dark green")
        self.scope.cam.setFluors(None)
        
        self._generate_and_set_fluorophores()
        #event.Skip()

    def OnBSavePointsButton(self, event):
        fn = wx.SaveFileSelector('Save point positions to file', '.txt')
        if fn is None:
            logger.warning('No file selected, cancelling')
            return

        #self.points = pylab.load(fn)
        # pylab.save(fn, scipy.array(self.points))
        np.savetxt(fn, scipy.array(self.points))
        #self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        #event.Skip()
        
    def OnBSetPSFModel(self, event=None):
        psf_settings = PSFSettings()
        psf_settings.configure_traits(kind='modal')
        
        z_modes = {int(k):float(v) for k, v in psf_settings.zernike_modes.items()}
        
        if psf_settings.four_pi:
            z_modes_lower = {int(k): float(v) for k, v in psf_settings.zernike_modes_lower.items()}
            phases = [np.pi*float(p) for p in psf_settings.phases]
            
            #print z_modes, z_modes_lower, phases
            rend_im.genTheoreticalModel4Pi(rend_im.mdh, phases=phases, zernikes=[z_modes, z_modes_lower],
                                           lamb=psf_settings.wavelength_nm,
                                           NA=psf_settings.NA, vectorial=psf_settings.vectorial)
            
            self.st_psf.SetLabelText('PSF: 4Pi %s [%1.2f NA @ %d nm, zerns=%s]' % ('vectorial' if  psf_settings.vectorial else 'scalar',psf_settings.NA, psf_settings.wavelength_nm, z_modes))
        else:
            logger.info('Setting PSF with zernike modes: %s' % z_modes)
            rend_im.genTheoreticalModel(rend_im.mdh, zernikes=z_modes, lamb=psf_settings.wavelength_nm,
                                        NA=psf_settings.NA, vectorial=psf_settings.vectorial)

            self.st_psf.SetLabelText('PSF: Widefield %s [%1.2f NA @ %d nm, zerns=%s]' % ('vectorial' if  psf_settings.vectorial else 'scalar',psf_settings.NA, psf_settings.wavelength_nm, z_modes))

    def OnBSetPSF(self, event):
        fn = wx.FileSelector('Read PSF from file', default_extension='psf', wildcard='PYME PSF Files (*.psf)|*.psf|TIFF (*.tif)|*.tif')
        logger.debug('Setting PSF from file: %s' % fn)
        if fn == '':
            #rend_im.genTheoreticalModel(rend_im.mdh)
            return
        else:
            rend_im.setModel(fn, rend_im.mdh)
            self.st_psf.SetLabelText('PSF: Experimental [%s]' % fn)
        #event.Skip()
            
    def OnBSavePSF(self, event):
        from PYME.DSView import ViewIm3D
        
        #fn = wx.SaveFileSelector('Save PSF to file', '.tif')
        #if fn is None:
        #    print('No file selected')
        #    return

        ViewIm3D(rend_im.get_psf(), mode='psf')
        

    def OnBGenFloursButton(self, event):
        if (len(self.points) == 0):
            wx.MessageBox( 'No fluorophore positions - either generate of load a set of positions', 'Error',wx.OK | wx.ICON_HAND)
            return
        
        transTens = self.getTensorFromGrids()
        exCrosses = [float(self.tExSwitch.GetValue()), float(self.tExProbe.GetValue())]
        #fluors = [fluor.fluorophore(x, y, z, transTens, exCrosses, activeState=self.activeState) for (x,y,z) in self.points]
        points_a = scipy.array(self.points).astype('f')
        x = points_a[:,0]
        y = points_a[:,1]
        z = points_a[:,2]
        #fluors = fluor.fluors(x, y, z, transTens, exCrosses, activeState=self.activeState)

        if points_a.shape[1] == 4: #4th entry is index into spectrum table
            c = points_a[:,3].astype('i')
            spec_sig = scipy.ones((len(x), 2))
            spec_sig[:,0] = self.spectralSignatures[c, 0]
            spec_sig[:,1] = self.spectralSignatures[c, 1]            
        
            fluors = fluor.SpectralFluorophores(x, y, z, transTens, exCrosses, activeState=self.activeState, spectralSig=spec_sig)
        else:
            fluors = fluor.Fluorophores(x, y, z, transTens, exCrosses, activeState=self.activeState)

        chan_z_offsets, chan_specs = self.getSplitterInfo()
        self.scope.cam.setSplitterInfo(chan_z_offsets, chan_specs)
        
        self.scope.cam.setFluors(fluors)


    def _generate_and_set_fluorophores(self):
        if self.nSimulationType.GetCurrentPage() == self.pFirstPrinciples:
            self.OnBGenFloursButton(None)
        else:
            self.OnBGenEmpiricalHistFluorsButton(None)
        
#        pylab.figure(1)
#        pylab.clf()
#        pylab.scatter([p[0] for p in self.points],[p[1] for p in self.points], c = [p[2] for p in self.points], cmap=pylab.cm.gist_rainbow, hold=False)
#        pylab.gca().set_ylim(self.scope.cam.YVals[0], self.scope.cam.YVals[-1])
#        pylab.gca().set_xlim(self.scope.cam.XVals[0], self.scope.cam.XVals[-1])
#        pylab.axis('equal')
#        pylab.colorbar()
#        pylab.show()
        #event.Skip()

    def OnBPauseButton(self, event):
        if self.scope.frameWrangler.isRunning():
            self.scope.frameWrangler.stop()
            self.bPause.SetLabel('Resume')
        else:
            self.scope.frameWrangler.start()
            self.bPause.SetLabel('Pause')
        #event.Skip()

    def OnTRefreshTimer(self, event):
        cts = scipy.zeros((len(self.states)))
        #for f in self.scope.cam.fluors:
        #    cts[f.state] +=1
        if self.scope.cam.fluors is None:
           self.stStatus.SetLabel('No fluorophores defined') 
           return

        for i in range(len(cts)):
            cts[i] = (self.scope.cam.fluors.fl['state'] == i).sum()
        
        labStr = 'Total # of fluorophores = %d\n' % len(self.scope.cam.fluors.fl)
        for i in range(len(cts)):
            labStr += "Num '%s' = %d\n" % (self.states[i], cts[i]) 
        self.stStatus.SetLabel(labStr)
        #event.Skip()

    def OnNumChannelsChanged(self, event=None):
        n_chans = int(self.cNumSplitterChans.GetStringSelection()[0])

        
        #print n_chans
        
        for i in range(4):
            if i < n_chans:
                #enable
                for j in range(2):
                    self.gSplitter.SetReadOnly(j, i, False)
                    #self.gSplitter.SetCellBackgroundColour(i, j, wx.LIGHT_GREY)
                    self.gSplitter.SetCellTextColour(j, i, wx.BLACK)
            else:
                #disable
                for j in range(2):
                    self.gSplitter.SetReadOnly(j, i)
                    #self.gSplitter.SetCellBackgroundColour(i, j, wx.LIGHT_GREY)
                    self.gSplitter.SetCellTextColour(j, i, wx.LIGHT_GREY)
                    
        self.gSplitter.Refresh()
        
        try:
            self.scope.frameWrangler.stop()
            self.scope.cam.SetSensorDimensions(n_chans*len(self.scope.cam.YVals), len(self.scope.cam.YVals), self.scope.cam.pixel_size_nm)
            self.scope.frameWrangler.Prepare()
            self.scope.frameWrangler.start()
        except AttributeError:
            logger.exception('Error setting new camera dimensions')
            pass
        
    def OnModelPresets(self, event=None):
        model = self.cModelPresets.GetStringSelection()
        
        if (model == 'STORM'): #default, STORM, no bleaching
            dlg = STORMPresetDialog(self, title='Specify transition rates for STORM/dSTORM')
        elif (model == 'PALM'): # PALM
            dlg = PALMPresetDialog(self, title='Specify transition rates for PALM')
        else:
            dlg = PAINTPresetDialog(self, title='Specify transition rates for PAINT')
            
        if (dlg.ShowModal() == wx.ID_OK):
            trans_matrix = dlg.get_trans_tensor()

            self.fillGrids(trans_matrix)
        
        dlg.Destroy()
         
            
    
    def OnBLoadEmpiricalHistButton(self, event):
        from . import EmpiricalHist
        fn = wx.FileSelector('Read point positions from file')
        if fn is None:
            logger.warning('No file selected, cancelling')
            return

        with open(fn,'r') as f:
            data = json.load(f)
        self.EmpiricalHist = EmpiricalHist(**data[data.keys().pop()])

        self.stEmpiricalHist.SetLabel('File: %s' % fn)

    def OnBGenEmpiricalHistFluorsButton(self, event):
        points_a = scipy.array(self.points).astype('f')
        x = points_a[:, 0]
        y = points_a[:, 1]
        z = points_a[:, 2]

        fluors = fluor.EmpiricalHistFluors(x, y, z,
                                           histogram=self.EmpiricalHist,
                                           activeState=self.activeState)

        chan_z_offsets, chan_specs = self.getSplitterInfo()
        self.scope.cam.setSplitterInfo(chan_z_offsets, chan_specs)

        self.scope.cam.setFluors(fluors)


class PAINTPresetDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Unbinding rate [per s]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tOnDark = wx.TextCtrl(self, -1, '1.0')
        hsizer.Add(self.tOnDark, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Binding rate [per s]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL,
                   2)
        self.tDarkOn = wx.TextCtrl(self, -1, '0.001')
        hsizer.Add(self.tDarkOn, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(self, -1, 'UV induced Dark-On rate [per mWs]:'), 0,
        #            wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        # self.tDarkOnUV = wx.TextCtrl(self, -1, '0.001')
        # hsizer.Add(self.tDarkOnUV, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        # sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(self, -1, 'Bleaching rate [per mWs]:'), 0,
        #            wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        # self.tOnBleach = wx.TextCtrl(self, -1, '0')
        # hsizer.Add(self.tOnBleach, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        # sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        btnsizer = self.CreateButtonSizer(wx.OK)#wx.StdDialogButtonSizer()

        #self.bOK = wx.Button(self, wx.ID_OK, "OK")
        #btnsizer.AddButton(self.bOK)

        sizer.Add(btnsizer, 0, wx.TOP | wx.EXPAND, 2)
        
        self.SetSizerAndFit(sizer)
    
    def get_trans_tensor(self):
        return fluor.createSimpleTransitionMatrix(pPA=[0, 0, 0],
                                                  pOnDark=[float(self.tOnDark.GetValue()), 0, 0],
                                                  pDarkOn=[float(self.tDarkOn.GetValue()),0, 0],
                                                  pOnBleach=[0, 0, 0],
                                                  pCagedBlinked=[1e9,0,0])


class STORMPresetDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'On-Dark rate [per mWs]:'), 0, wx.ALL|wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tOnDark = wx.TextCtrl(self,-1, '0.1')
        hsizer.Add(self.tOnDark, 0, wx.ALL|wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Spontaneous Dark-On rate [per s]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tDarkOn = wx.TextCtrl(self, -1, '0.001')
        hsizer.Add(self.tDarkOn, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'UV induced Dark-On rate [per mWs]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL,2)
        self.tDarkOnUV = wx.TextCtrl(self, -1, '0.001')
        hsizer.Add(self.tDarkOnUV, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Bleaching rate [per mWs]:'), 0,
                   wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tOnBleach = wx.TextCtrl(self, -1, '0.03')
        hsizer.Add(self.tOnBleach, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        btnsizer = self.CreateButtonSizer(wx.OK)#wx.StdDialogButtonSizer()
        
        #self.bOK = wx.Button(self, wx.ID_OK, "OK")
        #btnsizer.AddButton(self.bOK)
        
        sizer.Add(btnsizer, 0, wx.TOP|wx.EXPAND, 2)
        
        self.SetSizerAndFit(sizer)
        
    def get_trans_tensor(self):
        return fluor.createSimpleTransitionMatrix(pPA=[1e9,0,0] ,
                                                  pOnDark=[0,0,float(self.tOnDark.GetValue())],
                                                  pDarkOn=[float(self.tDarkOn.GetValue()),float(self.tDarkOnUV.GetValue()),0],
                                                  pOnBleach=[0,0,float(self.tOnBleach.GetValue())])


class PALMPresetDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        
        sizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Photoactivation rate [per mWs]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tPhotoactivation = wx.TextCtrl(self, -1, '0.001')
        hsizer.Add(self.tPhotoactivation, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Photoactivation rate (readout laser) [per mWs]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tPhotoactivationReadout = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.tPhotoactivationReadout, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Bleaching rate [per mWs]:'), 0,
                   wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tOnBleach = wx.TextCtrl(self, -1, '0.03')
        hsizer.Add(self.tOnBleach, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'On-Dark rate (blinking) [per mWs]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tOnDark = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.tOnDark, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Spontaneous Dark-On rate (blinking) [per s]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL,
                   2)
        self.tDarkOn = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.tDarkOn, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'UV induced Dark-On rate (blinking) [per mWs]:'), 0,
                   wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tDarkOnUV = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.tDarkOnUV, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)

        btnsizer = self.CreateButtonSizer(wx.OK)#wx.StdDialogButtonSizer()

        #self.bOK = wx.Button(self, wx.ID_OK, "OK")
        #btnsizer.AddButton(self.bOK)

        sizer.Add(btnsizer, 0, wx.TOP | wx.EXPAND, 2)
        
        self.SetSizerAndFit(sizer)
    
    def get_trans_tensor(self):
        return fluor.createSimpleTransitionMatrix(pPA=[0, float(self.tPhotoactivation.GetValue()), float(self.tPhotoactivationReadout.GetValue())],
                                                  pOnDark=[0, 0, float(self.tOnDark.GetValue())],
                                                  pDarkOn=[float(self.tDarkOn.GetValue()),
                                                           float(self.tDarkOnUV.GetValue()), 0],
                                                  pOnBleach=[0, 0, float(self.tOnBleach.GetValue())])