#Boa:Dialog:dSimControl

import wx
import wx.grid
import fluor
import wormlike2
import pylab
import scipy
import os

def create(parent):
    return dSimControl(parent)

[wxID_DSIMCONTROL, wxID_DSIMCONTROLBGENFLOURS, wxID_DSIMCONTROLBGENWORMLIKE, 
 wxID_DSIMCONTROLBLOADPOINTS, wxID_DSIMCONTROLBPAUSE, 
 wxID_DSIMCONTROLBSAVEPOINTS, wxID_DSIMCONTROLCBFLATTEN, 
 wxID_DSIMCONTROLGPROBE, wxID_DSIMCONTROLGSPONTAN, wxID_DSIMCONTROLGSWITCH, 
 wxID_DSIMCONTROLNTRANSITIONTENSOR, wxID_DSIMCONTROLSTATICBOX1, 
 wxID_DSIMCONTROLSTATICBOX2, wxID_DSIMCONTROLSTATICBOX3, 
 wxID_DSIMCONTROLSTATICBOX4, wxID_DSIMCONTROLSTATICBOX5, 
 wxID_DSIMCONTROLSTATICTEXT1, wxID_DSIMCONTROLSTATICTEXT2, 
 wxID_DSIMCONTROLSTATICTEXT3, wxID_DSIMCONTROLSTATICTEXT4, 
 wxID_DSIMCONTROLSTATICTEXT5, wxID_DSIMCONTROLSTCUROBJPOINTS, 
 wxID_DSIMCONTROLSTSTATUS, wxID_DSIMCONTROLTEXPROBE, 
 wxID_DSIMCONTROLTEXSWITCH, wxID_DSIMCONTROLTKBP, 
 wxID_DSIMCONTROLTNUMFLUOROPHORES, 
] = [wx.NewId() for _init_ctrls in range(27)]

[wxID_DSIMCONTROLTREFRESH] = [wx.NewId() for _init_utils in range(1)]

class dSimControl(wx.Dialog):
    def _init_coll_nTransitionTensor_Pages(self, parent):
        # generated method, don't edit

        parent.AddPage(imageId=-1, page=self.gSpontan, select=False,
              text='Spontaneous')
        parent.AddPage(imageId=-1, page=self.gSwitch, select=False,
              text='Switching Laser')
        parent.AddPage(imageId=-1, page=self.gProbe, select=True,
              text='Probe Laser')

    def _init_utils(self):
        # generated method, don't edit
        self.tRefresh = wx.Timer(id=wxID_DSIMCONTROLTREFRESH, owner=self)
        self.Bind(wx.EVT_TIMER, self.OnTRefreshTimer,
              id=wxID_DSIMCONTROLTREFRESH)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Dialog.__init__(self, id=wxID_DSIMCONTROL, name='dSimControl',
              parent=prnt, pos=wx.Point(688, 108), size=wx.Size(441, 605),
              style=wx.DEFAULT_DIALOG_STYLE, title='PALM Sim Monitor')
        self._init_utils()
        self.SetClientSize(wx.Size(433, 578))

        self.staticBox1 = wx.StaticBox(id=wxID_DSIMCONTROLSTATICBOX1,
              label='Fluorophore Postions', name='staticBox1', parent=self,
              pos=wx.Point(10, 8), size=wx.Size(414, 100), style=0)

        self.tNumFluorophores = wx.TextCtrl(id=wxID_DSIMCONTROLTNUMFLUOROPHORES,
              name='tNumFluorophores', parent=self, pos=wx.Point(22, 31),
              size=wx.Size(48, 21), style=0, value='1000')

        self.staticText1 = wx.StaticText(id=wxID_DSIMCONTROLSTATICTEXT1,
              label='fluorophores distributed evenly along', name='staticText1',
              parent=self, pos=wx.Point(75, 35), size=wx.Size(179, 13),
              style=0)

        self.tKbp = wx.TextCtrl(id=wxID_DSIMCONTROLTKBP, name='tKbp',
              parent=self, pos=wx.Point(258, 31), size=wx.Size(47, 21), style=0,
              value='1000')

        self.staticText2 = wx.StaticText(id=wxID_DSIMCONTROLSTATICTEXT2,
              label='kbp', name='staticText2', parent=self, pos=wx.Point(313,
              34), size=wx.Size(17, 13), style=0)

        self.bGenWormlike = wx.Button(id=wxID_DSIMCONTROLBGENWORMLIKE,
              label='Generate', name='bGenWormlike', parent=self,
              pos=wx.Point(339, 31), size=wx.Size(75, 23), style=0)
        self.bGenWormlike.Bind(wx.EVT_BUTTON, self.OnBGenWormlikeButton,
              id=wxID_DSIMCONTROLBGENWORMLIKE)

        self.bLoadPoints = wx.Button(id=wxID_DSIMCONTROLBLOADPOINTS,
              label='Load From File', name='bLoadPoints', parent=self,
              pos=wx.Point(32, 66), size=wx.Size(75, 23), style=0)
        self.bLoadPoints.Bind(wx.EVT_BUTTON, self.OnBLoadPointsButton,
              id=wxID_DSIMCONTROLBLOADPOINTS)

        self.bSavePoints = wx.Button(id=wxID_DSIMCONTROLBSAVEPOINTS,
              label='Save To File', name='bSavePoints', parent=self,
              pos=wx.Point(142, 66), size=wx.Size(75, 23), style=0)
        self.bSavePoints.Bind(wx.EVT_BUTTON, self.OnBSavePointsButton,
              id=wxID_DSIMCONTROLBSAVEPOINTS)

        self.staticBox2 = wx.StaticBox(id=wxID_DSIMCONTROLSTATICBOX2,
              label='Generate Virtual Fluorophores', name='staticBox2',
              parent=self, pos=wx.Point(8, 120), size=wx.Size(416, 344),
              style=0)

        self.staticBox3 = wx.StaticBox(id=wxID_DSIMCONTROLSTATICBOX3,
              label='Transition Tensor', name='staticBox3', parent=self,
              pos=wx.Point(16, 144), size=wx.Size(400, 216), style=0)

        self.nTransitionTensor = wx.Notebook(id=wxID_DSIMCONTROLNTRANSITIONTENSOR,
              name='nTransitionTensor', parent=self, pos=wx.Point(24, 160),
              size=wx.Size(384, 184), style=0)
        self.nTransitionTensor.SetLabel('Transition Probabilites')

        self.gSpontan = wx.grid.Grid(id=wxID_DSIMCONTROLGSPONTAN,
              name='gSpontan', parent=self.nTransitionTensor, pos=wx.Point(0,
              0), size=wx.Size(376, 158), style=0)

        self.gSwitch = wx.grid.Grid(id=wxID_DSIMCONTROLGSWITCH, name='gSwitch',
              parent=self.nTransitionTensor, pos=wx.Point(0, 0),
              size=wx.Size(376, 158), style=0)

        self.gProbe = wx.grid.Grid(id=wxID_DSIMCONTROLGPROBE, name='gProbe',
              parent=self.nTransitionTensor, pos=wx.Point(0, 0),
              size=wx.Size(376, 158), style=0)

        self.staticBox4 = wx.StaticBox(id=wxID_DSIMCONTROLSTATICBOX4,
              label='Excitation Crossections', name='staticBox4', parent=self,
              pos=wx.Point(18, 362), size=wx.Size(398, 53), style=0)

        self.bGenFlours = wx.Button(id=wxID_DSIMCONTROLBGENFLOURS, label='Go',
              name='bGenFlours', parent=self, pos=wx.Point(184, 424),
              size=wx.Size(75, 23), style=0)
        self.bGenFlours.Bind(wx.EVT_BUTTON, self.OnBGenFloursButton,
              id=wxID_DSIMCONTROLBGENFLOURS)

        self.staticText3 = wx.StaticText(id=wxID_DSIMCONTROLSTATICTEXT3,
              label='Switching Laser:', name='staticText3', parent=self,
              pos=wx.Point(32, 387), size=wx.Size(78, 13), style=0)

        self.tExSwitch = wx.TextCtrl(id=wxID_DSIMCONTROLTEXSWITCH,
              name='tExSwitch', parent=self, pos=wx.Point(124, 383),
              size=wx.Size(44, 21), style=0, value='5')

        self.staticText4 = wx.StaticText(id=wxID_DSIMCONTROLSTATICTEXT4,
              label='/mWs       Probe Laser:', name='staticText4', parent=self,
              pos=wx.Point(175, 387), size=wx.Size(109, 13), style=0)

        self.tExProbe = wx.TextCtrl(id=wxID_DSIMCONTROLTEXPROBE,
              name='tExProbe', parent=self, pos=wx.Point(292, 383),
              size=wx.Size(44, 21), style=0, value='10')

        self.staticText5 = wx.StaticText(id=wxID_DSIMCONTROLSTATICTEXT5,
              label='/mWs', name='staticText5', parent=self, pos=wx.Point(344,
              387), size=wx.Size(32, 13), style=0)

        self.staticBox5 = wx.StaticBox(id=wxID_DSIMCONTROLSTATICBOX5,
              label='Status', name='staticBox5', parent=self, pos=wx.Point(8,
              472), size=wx.Size(416, 100), style=0)

        self.stStatus = wx.StaticText(id=wxID_DSIMCONTROLSTSTATUS,
              label='hello\nworld', name='stStatus', parent=self,
              pos=wx.Point(24, 488), size=wx.Size(304, 80), style=0)

        self.bPause = wx.Button(id=wxID_DSIMCONTROLBPAUSE, label='Pause',
              name='bPause', parent=self, pos=wx.Point(340, 537),
              size=wx.Size(75, 23), style=0)
        self.bPause.Bind(wx.EVT_BUTTON, self.OnBPauseButton,
              id=wxID_DSIMCONTROLBPAUSE)

        self.stCurObjPoints = wx.StaticText(id=wxID_DSIMCONTROLSTCUROBJPOINTS,
              label='Current object has 0 points', name='stCurObjPoints',
              parent=self, pos=wx.Point(248, 84), size=wx.Size(131, 13),
              style=0)

        self.cbFlatten = wx.CheckBox(id=wxID_DSIMCONTROLCBFLATTEN,
              label='flatten (set z to 0)', name='cbFlatten', parent=self,
              pos=wx.Point(248, 62), size=wx.Size(136, 13), style=0)
        self.cbFlatten.SetValue(False)

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
        
        self.spectralSignatures = scipy.array([[1, 0.3], [0, 1]])

        self.scope=scope
        self.points = []
        self.tRefresh.Start(200)
        

    def OnBGenWormlikeButton(self, event):
        kbp = float(self.tKbp.GetValue())
        numFluors = int(self.tNumFluorophores.GetValue())
        wc = wormlike2.fibre30nm(kbp, 10*kbp/numFluors)
        
        wc.xp -= wc.xp.mean()
        wc.yp -= wc.yp.mean()
        wc.zp -= wc.zp.mean()
        
        self.points = []
        for i in range(len(wc.xp)):
            if not self.cbFlatten.GetValue():
                #self.points.append((wc.xp[i],wc.yp[i],wc.zp[i], float(i > len(wc.xp)/2)))
                self.points.append((wc.xp[i],wc.yp[i],wc.zp[i]))
            else:
                #self.points.append((wc.xp[i],wc.yp[i],0,float(i > len(wc.xp)/2)))
                self.points.append((wc.xp[i],wc.yp[i],0))
        
        self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        #event.Skip()

    def OnBLoadPointsButton(self, event):
        fn = wx.FileSelector('Read point positions from file')
        if fn == None:
            print 'No file selected'
            return

        self.points = pylab.load(fn)

        self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
        #event.Skip()

    def OnBSavePointsButton(self, event):
        fn = wx.SaveFileSelector('Save point positions to file', '.txt')
        if fn == None:
            print 'No file selected'
            return

        #self.points = pylab.load(fn)
        pylab.save(fn, scipy.array(self.points))
        #self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.points))
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

        
        self.scope.cam.fluors=fluors
        
        pylab.figure(1)
        pylab.plot([p[0] for p in self.points],[p[1] for p in self.points], '.', hold=False)
        pylab.gca().set_ylim(self.scope.cam.YVals[-1], self.scope.cam.YVals[0])
        pylab.gca().set_xlim(self.scope.cam.XVals[0], self.scope.cam.XVals[-1])
        pylab.show()
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
