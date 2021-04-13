import wx
import wx.grid
import PYME.ui.manualFoldPanel as afp

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

import numpy as np
from . import fluor
from . import simcontrol

import logging
logger = logging.getLogger(__name__)

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
        self._init_utils()
        
        ################ Splitter ######################
        
        item = afp.foldingPane(self, -1, caption="Virtual Hardware", pinned=True)
        
        
        pane = wx.Panel(item, -1)
        
        sbsizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(pane, -1, 'Number of detection channels: '), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        self.cNumSplitterChans = wx.Choice(pane, -1,
                                           choices=['1 - Standard', '2 - Ratiometric/Biplane', '4 - HT / 4Pi-SMS'])
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

        self.bViewPSF = wx.Button(pane, -1, 'View')
        self.bViewPSF.Bind(wx.EVT_BUTTON, self.OnBViewPSF)
        hsizer.Add(self.bViewPSF, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        
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
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.tNumFluorophores = wx.TextCtrl(pane, -1, value='10000', size=(60, -1))
        hsizer.Add(self.tNumFluorophores, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        hsizer.Add(wx.StaticText(pane, -1, 'fluorophores distributed evenly along'),
                   0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        self.tKbp = wx.TextCtrl(pane, -1, size=(60, -1), value='200000')
        hsizer.Add(self.tKbp, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        hsizer.Add(wx.StaticText(pane, -1, 'nm'),
                   0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        hsizer.AddStretchSpacer()
        
        self.bGenWormlike = wx.Button(pane, -1, 'Generate')
        self.bGenWormlike.Bind(wx.EVT_BUTTON, self.OnBGenWormlikeButton)
        hsizer.Add(self.bGenWormlike, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(pane, -1, 'Persistence length [nm]:'),
                   0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        self.tPersist = wx.TextCtrl(pane, -1, size=(60, -1), value='1500')
        hsizer.Add(self.tPersist, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        hsizer.Add(wx.StaticText(pane, -1, 'Z scale:'),
                   0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        self.tZScale = wx.TextCtrl(pane, -1, size=(60, -1), value='1.0')
        hsizer.Add(self.tZScale, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        self.cbFlatten = wx.CheckBox(pane, -1, 'flatten (set z to 0)')
        self.cbFlatten.SetValue(False)
        hsizer.Add(self.cbFlatten, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.cbColour = wx.CheckBox(pane, -1, u'Colourful')
        self.cbColour.SetValue(False)
        hsizer.Add(self.cbColour, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        self.cbWrap = wx.CheckBox(pane, -1, u'Wrap at FOV edge')
        self.cbWrap.SetValue(True)
        hsizer.Add(self.cbWrap, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.stCurObjPoints = wx.StaticText(pane, -1, 'Current object has NO points')
        self.stCurObjPoints.SetForegroundColour(wx.RED)
        hsizer.Add(self.stCurObjPoints, 0, wx.ALL, 2)
        hsizer.AddStretchSpacer()
        
        self.bLoadPoints = wx.Button(pane, -1, 'Load From File')
        self.bLoadPoints.Bind(wx.EVT_BUTTON, self.OnBLoadPointsButton)
        hsizer.Add(self.bLoadPoints, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        self.bSavePoints = wx.Button(pane, -1, 'Save To File')
        self.bSavePoints.Bind(wx.EVT_BUTTON, self.OnBSavePointsButton)
        hsizer.Add(self.bSavePoints, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        
        sbsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
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
        
        hsizer.Add(wx.StaticText(pFirstPrinciples, -1, 'Generate matrix for:'), 0, wx.ALL,
                   2)
        
        self.cModelPresets = wx.Choice(pFirstPrinciples, -1, choices=['STORM', 'PALM', 'PAINT'])
        self.cModelPresets.Bind(wx.EVT_CHOICE, self.OnModelPresets)
        
        hsizer.Add(self.cModelPresets, 0, wx.ALL, 2)
        
        sbsizer2.Add(hsizer, 0, wx.ALL, 2)
        
        self.gSpontan = wx.grid.Grid(self.nTransitionTensor, -1)
        self.gSwitch = wx.grid.Grid(self.nTransitionTensor, -1)
        self.gProbe = wx.grid.Grid(self.nTransitionTensor, -1)
        
        self._init_coll_nTransitionTensor_Pages(self.nTransitionTensor)
        
        self.setupGrid(self.gSpontan, self.sim_controller.states, self.sim_controller.stateTypes)
        self.setupGrid(self.gSwitch, self.sim_controller.states, self.sim_controller.stateTypes)
        self.setupGrid(self.gProbe, self.sim_controller.states, self.sim_controller.stateTypes)
        
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
                     wx.ALIGN_CENTER_VERTICAL, 2)
        
        pEmpiricalModelSizer.Add(sbsizer2, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.bGenEmpiricalHistFluors = wx.Button(pEmpiricalModel, -1, 'Go')
        self.bGenEmpiricalHistFluors.Bind(wx.EVT_BUTTON, self.OnBGenEmpiricalHistFluorsButton)
        hsizer.Add(self.bGenEmpiricalHistFluors, 1,
                   wx.ALIGN_CENTER_VERTICAL, 2)
        
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
        sbsizer.Add(self.stStatus, 0, wx.ALL | wx.EXPAND, 2)
        
        self.bPause = wx.Button(pane, -1, 'Pause')
        self.bPause.Bind(wx.EVT_BUTTON, self.OnBPauseButton)
        sbsizer.Add(self.bPause, 0, wx.ALL | wx.ALIGN_RIGHT, 2)
        
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
        nStates = len(self.sim_controller.states)
        for i in range(nStates):
            for j in range(nStates):
                self.gSpontan.SetCellValue(i, j, '%f' % vals[i, j, 0])
                self.gSwitch.SetCellValue(i, j, '%f' % vals[i, j, 1])
                self.gProbe.SetCellValue(i, j, '%f' % vals[i, j, 2])
    
    def getTensorFromGrids(self):
        nStates = len(self.sim_controller.states)
        transTens = np.zeros((nStates, nStates, 3))
        
        for i in range(nStates):
            for j in range(nStates):
                transTens[i, j, 0] = float(self.gSpontan.GetCellValue(i, j))
                transTens[i, j, 1] = float(self.gSwitch.GetCellValue(i, j))
                transTens[i, j, 2] = float(self.gProbe.GetCellValue(i, j))
        
        return transTens
    
    def setupSplitterGrid(self):
        grid = self.gSplitter
        
        #grid.SetDefaultColSize(70)
        grid.CreateGrid(2, 4)
        
        grid.SetRowLabelValue(0, 'Spectral chan')
        grid.SetRowLabelValue(1, 'Z offset [nm]')
        
        #hard code some initial values
        z_offset, spec = self.sim_controller.z_offsets, self.sim_controller.spec_chans
        
        for i in range(4):
            grid.SetColLabelValue(i, 'Chan %d' % i)
            
            grid.SetCellValue(0, i, '%d' % spec[i])
            grid.SetCellValue(1, i, '%3.2f' % z_offset[i])
            
            #grid.SetCellTextColour(i, i, wx.LIGHT_GREY)
    
    def getSplitterInfo(self):
        nChans = int(self.cNumSplitterChans.GetStringSelection()[0])
        
        self.sim_controller.z_offsets = [float(self.gSplitter.GetCellValue(1, i)) for i in range(nChans)]
        self.sim_controller.spec_chans = [int(self.gSplitter.GetCellValue(0, i)) for i in range(nChans)]
        
        self.sim_controller.n_chans = nChans
        
    
    def __init__(self, parent, sim_controller):
        afp.foldPanel.__init__(self, parent, -1)
        self.sim_controller = sim_controller # type: .simcontrol.SimController
        
        self._init_ctrls(parent)
        
        self.fillGrids(self.sim_controller.transition_tensor)
        
        self.tRefresh.Start(200)

    
    def OnBGenWormlikeButton(self, event):
        kbp = float(self.tKbp.GetValue())
        numFluors = int(self.tNumFluorophores.GetValue())
        persistLength = float(self.tPersist.GetValue())
        
        num_colours = len(self.sim_controller.spectralSignatures) if self.cbColour.GetValue() else 1

        self.sim_controller.gen_fluors_wormlike(kbp, persistLength, numFluors,
                                                flatten=self.cbFlatten.GetValue(), wrap=self.cbWrap.GetValue(),
                                                z_scale = float(self.tZScale.GetValue()),num_colours= num_colours)
        
        
        self.stCurObjPoints.SetLabel('Current object has %d points' % len(self.sim_controller.points))
        self.stCurObjPoints.SetForegroundColour("dark green")
        
    
    def OnBLoadPointsButton(self, event):
        fn = wx.FileSelector('Read point positions from file')
        if fn is None:
            logger.warning('No file selected')
        else:
            self.sim_controller.load_fluors(fn)
        
    
    def OnBSavePointsButton(self, event):
        fn = wx.SaveFileSelector('Save point positions to file', '.txt')
        if fn is None:
            logger.warning('No file selected')
        else:
            self.sim_controller.save_points(fn)
  
    
    def OnBSetPSFModel(self, event=None):
        psf_settings = simcontrol.PSFSettings()
        psf_settings.configure_traits(kind='modal')

        self.st_psf.SetLabelText(self.sim_controller.set_psf_model(psf_settings))
    
    def OnBSetPSF(self, event):
        fn = wx.FileSelector('Read PSF from file', default_extension='psf',
                             wildcard='PYME PSF Files (*.psf)|*.psf|TIFF (*.tif)|*.tif')
        logger.debug('Setting PSF from file: %s' %fn)
        if fn == '':
            return
        else:
            self.st_psf.SetLabelText(self.sim_controller.set_psf_from_file(fn))

    def OnBViewPSF(self, event):
        from PYME.DSView import ViewIm3D
    
        #fn = wx.SaveFileSelector('Save PSF to file', '.tif')
        #if fn is None:
        #    print('No file selected')
        #    return
    
        ViewIm3D(self.sim_controller.get_psf(), mode='psf')
    
    def OnBGenFloursButton(self, event):
        if (len(self.sim_controller.points) == 0):
            wx.MessageBox('No fluorophore positions - either generate of load a set of positions', 'Error',
                          wx.OK | wx.ICON_HAND)
            return
        
        self.sim_controller.transition_tensor = self.getTensorFromGrids()
        self.sim_controller.excitation_crossections = [float(self.tExSwitch.GetValue()), float(self.tExProbe.GetValue())]
        self.getSplitterInfo()
        
        self.sim_controller.generate_fluorophores_theoretical()
    
    def _generate_and_set_fluorophores(self):
        self.getSplitterInfo()
        if self.nSimulationType.GetCurrentPage() == self.pFirstPrinciples:
            self.sim_controller.transition_tensor = self.getTensorFromGrids()
            self.sim_controller.excitation_crossections = [float(self.tExSwitch.GetValue()),
                                                           float(self.tExProbe.GetValue())]
            self.sim_controller.generate_fluorophores_theoretical()
        else:
            self.sim_controller.generate_fluorophores_empirical()
        
    
    def OnBPauseButton(self, event):
        if self.sim_controller.scope.frameWrangler.isRunning():
            self.sim_controller.scope.frameWrangler.stop()
            self.bPause.SetLabel('Resume')
        else:
            self.sim_controller.scope.frameWrangler.start()
            self.bPause.SetLabel('Pause')
            #event.Skip()
    
    def OnTRefreshTimer(self, event):
        cts = np.zeros((len(self.sim_controller.states)))
        #for f in self.scope.cam.fluors:
        #    cts[f.state] +=1
        if self.sim_controller.scope.cam.fluors is None:
            self.stStatus.SetLabel('No fluorophores defined')
            return
        
        for i in range(len(cts)):
            cts[i] = (self.sim_controller.scope.cam.fluors.fl['state'] == i).sum()
        
        labStr = 'Total # of fluorophores = %d\n' % len(self.sim_controller.scope.cam.fluors.fl)
        for i in range(len(cts)):
            labStr += "Num '%s' = %d\n" % (self.sim_controller.states[i], cts[i])
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
        
        self.sim_controller.change_num_channels(n_chans)
        if (len(self.sim_controller.points) > 0):
            self._generate_and_set_fluorophores()
    
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
        fn = wx.FileSelector('Read point positions from file')
        if fn is None:
            logger.warning('No file selected')
        else:
            self.sim_controller.load_empirical_histogram(fn)
            self.stEmpiricalHist.SetLabel('File: %s' % fn)
    
    def OnBGenEmpiricalHistFluorsButton(self, event):
        self.sim_controller.generate_fluorophores_empirical()


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
                                                  pDarkOn=[float(self.tDarkOn.GetValue()), 0, 0],
                                                  pOnBleach=[0, 0, 0],
                                                  pCagedBlinked=[1e9, 0, 0])


class STORMPresetDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'On-Dark rate [per mWs]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tOnDark = wx.TextCtrl(self, -1, '0.1')
        hsizer.Add(self.tOnDark, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Spontaneous Dark-On rate [per s]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL,
                   2)
        self.tDarkOn = wx.TextCtrl(self, -1, '0.001')
        hsizer.Add(self.tDarkOn, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'UV induced Dark-On rate [per mWs]:'), 0,
                   wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
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
        
        sizer.Add(btnsizer, 0, wx.TOP | wx.EXPAND, 2)
        
        self.SetSizerAndFit(sizer)
    
    def get_trans_tensor(self):
        return fluor.createSimpleTransitionMatrix(pPA=[1e9, 0, 0],
                                                  pOnDark=[0, 0, float(self.tOnDark.GetValue())],
                                                  pDarkOn=[float(self.tDarkOn.GetValue()),
                                                           float(self.tDarkOnUV.GetValue()), 0],
                                                  pOnBleach=[0, 0, float(self.tOnBleach.GetValue())])


class PALMPresetDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Photoactivation rate [per mWs]:'), 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL,
                   2)
        self.tPhotoactivation = wx.TextCtrl(self, -1, '0.001')
        hsizer.Add(self.tPhotoactivation, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Photoactivation rate (readout laser) [per mWs]:'), 0,
                   wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
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
        hsizer.Add(wx.StaticText(self, -1, 'On-Dark rate (blinking) [per mWs]:'), 0,
                   wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        self.tOnDark = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.tOnDark, 0, wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL, 2)
        sizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 2)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Spontaneous Dark-On rate (blinking) [per s]:'), 0,
                   wx.ALL | wx.ALIGN_CENTRE_HORIZONTAL,
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
        return fluor.createSimpleTransitionMatrix(
            pPA=[0, float(self.tPhotoactivation.GetValue()), float(self.tPhotoactivationReadout.GetValue())],
            pOnDark=[0, 0, float(self.tOnDark.GetValue())],
            pDarkOn=[float(self.tDarkOn.GetValue()),
                     float(self.tDarkOnUV.GetValue()), 0],
            pOnBleach=[0, 0, float(self.tOnBleach.GetValue())])
