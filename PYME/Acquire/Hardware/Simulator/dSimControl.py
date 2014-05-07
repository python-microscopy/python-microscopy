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

import wx
import wx.grid
from . import fluor
from . import wormlike2
import pylab
import scipy
import numpy as np
#import os
from . import rend_im

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

class dSimControl(wx.Panel):
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
        wx.Panel.__init__(self, id=-1, parent=prnt)#, size=wx.Size(442, 637))
        self._init_utils()
        #self.SetClientSize(wx.Size(434, 610))
        vsizer= wx.BoxSizer(wx.VERTICAL)


        ########### Fluorophore Positions ############        
        sbsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Fluorophore Postions'), 
                                  wx.VERTICAL)        
        hsizer=wx.BoxSizer(wx.HORIZONTAL)

        self.tNumFluorophores = wx.TextCtrl(self, -1, value='10000', size=(60, -1))
        hsizer.Add(self.tNumFluorophores, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        hsizer.Add(wx.StaticText(self,-1,'fluorophores distributed evenly along'), 
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.tKbp = wx.TextCtrl(self, -1, size=(60, -1), value='200000')
        hsizer.Add(self.tKbp, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        hsizer.Add(wx.StaticText(self,-1,'nm'), 
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
                   
        hsizer.AddStretchSpacer()

        self.bGenWormlike = wx.Button(self, -1,'Generate')
        self.bGenWormlike.Bind(wx.EVT_BUTTON, self.OnBGenWormlikeButton)
        hsizer.Add(self.bGenWormlike, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 2)
        hsizer=wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self,-1,'Persistence length [nm]:'), 
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        self.tPersist = wx.TextCtrl(self, -1, size=(60, -1), value='1500')
        hsizer.Add(self.tPersist, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        hsizer.Add(wx.StaticText(self,-1,'Z scale:'), 
                   0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        self.tZScale = wx.TextCtrl(self, -1, size=(60, -1), value='1.0')
        hsizer.Add(self.tZScale, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.cbFlatten = wx.CheckBox(self, -1, 'flatten (set z to 0)')
        self.cbFlatten.SetValue(False)
        hsizer.Add(self.cbFlatten, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.cbColour = wx.CheckBox(self, -1, u'colourful')
        self.cbColour.SetValue(False)
        hsizer.Add(self.cbColour, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        hsizer=wx.BoxSizer(wx.HORIZONTAL)
        
        self.stCurObjPoints = wx.StaticText(self, -1, 'Current object has 0 points')
        hsizer.Add(self.stCurObjPoints, 0, wx.ALL, 2)
        hsizer.AddStretchSpacer()

        self.bLoadPoints = wx.Button(self, -1,'Load From File')
        self.bLoadPoints.Bind(wx.EVT_BUTTON, self.OnBLoadPointsButton)
        hsizer.Add(self.bLoadPoints, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.bSavePoints = wx.Button(self, -1,'Save To File')
        self.bSavePoints.Bind(wx.EVT_BUTTON, self.OnBSavePointsButton)
        hsizer.Add(self.bSavePoints, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 2)

                
        
        
        
        vsizer.Add(sbsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        ################ Virtual Fluorophores ###########
        
        sbsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Generate Virtual Fluorophores'), 
                                  wx.VERTICAL)
                                  
        sbsizer2=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Transition Tensor'), 
                                  wx.VERTICAL)

        self.nTransitionTensor = wx.Notebook(self, -1)
        #self.nTransitionTensor.SetLabel('Transition Probabilites')

        self.gSpontan = wx.grid.Grid(self.nTransitionTensor, -1)
        self.gSwitch = wx.grid.Grid(self.nTransitionTensor, -1)
        self.gProbe = wx.grid.Grid(self.nTransitionTensor, -1)
        
        sbsizer2.Add(self.nTransitionTensor, 1, wx.EXPAND|wx.ALL, 2)
        sbsizer.Add(sbsizer2, 1, wx.EXPAND|wx.ALL, 2)
        
        sbsizer2=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Excitation Crossections'), 
                                  wx.HORIZONTAL)
                                  
        sbsizer2.Add(wx.StaticText(self, -1, 'Switching Laser:'), 0, 
                     wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.tExSwitch = wx.TextCtrl(self, -1, value='1')
        sbsizer2.Add(self.tExSwitch, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        sbsizer2.Add(wx.StaticText(self, -1, '/mWs       Probe Laser:'), 0, 
                     wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)

        self.tExProbe = wx.TextCtrl(self, -1, value='100')
        sbsizer2.Add(self.tExProbe, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        
        sbsizer2.Add(wx.StaticText(self, -1, '/mWs'), 0, 
                     wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
                                  
        sbsizer.Add(sbsizer2, 0, wx.EXPAND|wx.ALL, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL) 

        self.bSetPSF = wx.Button(self, -1,'Set PSF')
        self.bSetPSF.Bind(wx.EVT_BUTTON, self.OnBSetPSF)
        hsizer.Add(self.bSetPSF, 1, wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 2)                         

        self.bGenFlours = wx.Button(self, -1,'Go')
        self.bGenFlours.Bind(wx.EVT_BUTTON, self.OnBGenFloursButton)
        hsizer.Add(self.bGenFlours, 1, wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 2)

        
        
        sbsizer.Add(hsizer, 0, wx.ALL|wx.ALIGN_RIGHT, 2)
        vsizer.Add(sbsizer, 0, wx.ALL|wx.EXPAND, 2)

        ######## Status #########
        
        sbsizer=wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Status'), 
                                  wx.VERTICAL)

        self.stStatus = wx.StaticText(self, -1,
              label='hello\nworld\n\n\nfoo')
        sbsizer.Add(self.stStatus, 0, wx.ALL|wx.EXPAND, 2)

        self.bPause = wx.Button(self, -1,'Pause')
        self.bPause.Bind(wx.EVT_BUTTON, self.OnBPauseButton)
        sbsizer.Add(self.bPause, 0, wx.ALL|wx.ALIGN_RIGHT, 2)
        
        vsizer.Add(sbsizer, 0, wx.ALL|wx.EXPAND, 2)
        
        self.vsizer = vsizer



        self._init_coll_nTransitionTensor_Pages(self.nTransitionTensor)

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
        
    
    def __init__(self, parent, scope=None, states=['Caged', 'On', 'Blinked', 'Bleached'], stateTypes=[fluor.FROM_ONLY, fluor.ALL_TRANS, fluor.ALL_TRANS, fluor.TO_ONLY], startVals=None, activeState=fluor.states.active):
        self._init_ctrls(parent)
        
        self.states = states
        self.stateTypes = stateTypes
        self.activeState = activeState
        
        self.setupGrid(self.gSpontan, states, stateTypes)
        self.setupGrid(self.gSwitch, states, stateTypes)
        self.setupGrid(self.gProbe, states, stateTypes)
        
        if (startVals == None): #use defaults
            startVals = fluor.createSimpleTransitionMatrix()
            
        self.fillGrids(startVals)
        
        self.spectralSignatures = scipy.array([[1, 0.3], [.7, .7], [0.2, 1]])

        self.scope=scope
        self.points = []
        self.tRefresh.Start(200)
        self.SetSizerAndFit(self.vsizer)
        

    def OnBGenWormlikeButton(self, event):
        import numpy as np
        kbp = float(self.tKbp.GetValue())
        numFluors = int(self.tNumFluorophores.GetValue())
        persistLength= float(self.tPersist.GetValue())
        #wc = wormlike2.fibre30nm(kbp, 10*kbp/numFluors)
        wc = wormlike2.wiglyFibre(kbp, persistLength, kbp/numFluors)
        
        wc.xp -= wc.xp.mean() + 128*70
        wc.xp = np.mod(wc.xp, 256*70) - 128*70
        wc.yp -= wc.yp.mean() + 128*70
        wc.yp = np.mod(wc.yp, 256*70) - 128*70
        wc.zp -= wc.zp.mean()
        
        wc.zp *= float(self.tZScale.GetValue())
        
        self.points = []
        
        for i in range(len(wc.xp)):
            if not self.cbFlatten.GetValue():
                if self.cbColour.GetValue():
                    self.points.append((wc.xp[i],wc.yp[i],wc.zp[i], float(i/((len(wc.xp) + 1)/3))))
                else:
                    self.points.append((wc.xp[i],wc.yp[i],wc.zp[i]))
            else:
                if self.cbColour.GetValue():
                    self.points.append((wc.xp[i],wc.yp[i],0,float(i/((len(wc.xp)+1)/3))))
                else:
                    self.points.append((wc.xp[i],wc.yp[i],0))
        
        self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        #event.Skip()

    def OnBLoadPointsButton(self, event):
        fn = wx.FileSelector('Read point positions from file')
        if fn == None:
            print('No file selected')
            return

        self.points = np.loadtxt(fn)

        self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        #event.Skip()

    def OnBSavePointsButton(self, event):
        fn = wx.SaveFileSelector('Save point positions to file', '.txt')
        if fn == None:
            print('No file selected')
            return

        #self.points = pylab.load(fn)
        pylab.save(fn, scipy.array(self.points))
        #self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        #event.Skip()

    def OnBSetPSF(self, event):
        fn = wx.FileSelector('Read PSF from file', default_extension='psf', wildcard='PYME PSF Files (*.psf)|*.psf')
        print(fn)
        if fn == '':
            rend_im.genTheoreticalModel(rend_im.MetaData.TIRFDefault)
            return
        else:
            rend_im.setModel(fn, rend_im.MetaData.TIRFDefault)
        #event.Skip()

    def OnBGenFloursButton(self, event):
        transTens = self.getTensorFromGrids()
        exCrosses = [float(self.tExSwitch.GetValue()), float(self.tExProbe.GetValue())]
        #fluors = [fluor.fluorophore(x, y, z, transTens, exCrosses, activeState=self.activeState) for (x,y,z) in self.points]
        points_a = scipy.array(self.points)
        x = points_a[:,0]
        y = points_a[:,1]
        z = points_a[:,2]
        #fluors = fluor.fluors(x, y, z, transTens, exCrosses, activeState=self.activeState)

        if points_a.shape[1] == 4: #4th entry is index into spectrum table
            c = points_a[:,3].astype('i')
            spec_sig = scipy.ones((len(x), 2))
            spec_sig[:,0] = self.spectralSignatures[c, 0]
            spec_sig[:,1] = self.spectralSignatures[c, 1]            
        
            fluors = fluor.specFluors(x, y, z, transTens, exCrosses, activeState=self.activeState, spectralSig=spec_sig)
        else:
            fluors = fluor.fluors(x, y, z, transTens, exCrosses, activeState=self.activeState)

        
        self.scope.cam.setFluors(fluors)
        
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
        if self.scope.pa.isRunning():
            self.scope.pa.stop()
            self.bPause.SetLabel('Resume')
        else:
            self.scope.pa.start()
            self.bPause.SetLabel('Pause')
        #event.Skip()

    def OnTRefreshTimer(self, event):
        cts = scipy.zeros((len(self.states)))
        #for f in self.scope.cam.fluors:
        #    cts[f.state] +=1
        if self.scope.cam.fluors == None:
           self.stStatus.SetLabel('No fluorophores defined') 
           return

        for i in range(len(cts)):
            cts[i] = (self.scope.cam.fluors.fl['state'] == i).sum()
        
        labStr = 'Total # of fluorophores = %d\n' % len(self.scope.cam.fluors.fl)
        for i in range(len(cts)):
            labStr += "Num '%s' = %d\n" % (self.states[i], cts[i]) 
        self.stStatus.SetLabel(labStr)
        #event.Skip()
