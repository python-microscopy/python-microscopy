#!/usr/bin/python
##################
# LMAnalysis.py
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
import wx
import Pyro.core
from PYME.Analysis import remFitBuf
import os
from PYME.Acquire import MetaDataHandler
from pylab import *
from PYME.FileUtils import fileID
from PYME.FileUtils.nameUtils import genResultFileName
from PYME.Analysis.LMVis import progGraph as progGraph
from PYME.ParallelTasks.relativeFiles import getRelFilename
import glob
import numpy
import pylab
from PYME.FileUtils import nameUtils
import PYME.misc.autoFoldPanel as afp
from PYME.Acquire.mytimer import mytimer
from PYME.DSView import fitInfo
from PYME.DSView.OverlaysPanel import OverlayPanel
import wx.lib.agw.aui as aui

debug = True

def debugPrint(msg):
    if debug:
        print msg

class LMAnalyser:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        if 'tq' in dir(dsviewer):
            self.tq = dsviewer.tq
        else:
            self.tq = None
        
        self.image = dsviewer.image
        self.view = dsviewer.view
        self.do = dsviewer.do

        #this should only occur for files types which we weren't expecting to process
        #as LM data (eg tiffs)
        if not 'EstimatedLaserOnFrameNo' in self.image.mdh.getEntryNames():
            from PYME.Analysis import MetaData
            MetaData.fillInBlanks(self.image.mdh, self.image.dataSource)

        if 'fitResults' in dir(self.image):
            self.fitResults = self.image.fitResults
        else:
            self.fitResults = []
        
        if 'resultsMdh' in dir(self.image):
            self.resultsMdh = self.image.resultsMdh

        mTasks = wx.Menu()
        TASKS_STANDARD_2D = wx.NewId()
        TASKS_CALIBRATE_SPLITTER = wx.NewId()
        TASKS_2D_SPLITTER = wx.NewId()
        TASKS_3D = wx.NewId()
        TASKS_3D_SPLITTER = wx.NewId()
        TASKS_PRI = wx.NewId()
        mTasks.Append(TASKS_STANDARD_2D, "Normal 2D analysis", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_CALIBRATE_SPLITTER, "Calibrating the splitter", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_2D_SPLITTER, "2D with splitter", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_3D, "3D analysis", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_3D_SPLITTER, "3D with splitter", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_PRI, "PRI", "", wx.ITEM_NORMAL)
        self.dsviewer.menubar.Append(mTasks, "Set defaults for")
        
        wx.EVT_MENU(self.dsviewer, TASKS_CALIBRATE_SPLITTER, self.OnCalibrateSplitter)
        wx.EVT_MENU(self.dsviewer, TASKS_STANDARD_2D, self.OnStandard2D)
        wx.EVT_MENU(self.dsviewer, TASKS_2D_SPLITTER, self.OnSpitter2D)
        wx.EVT_MENU(self.dsviewer, TASKS_3D, self.OnStandard3D)
        wx.EVT_MENU(self.dsviewer, TASKS_3D_SPLITTER, self.OnSpliter3D)
        wx.EVT_MENU(self.dsviewer, TASKS_PRI, self.OnPRI3D)
        
        BG_SUBTRACT = wx.NewId()
        self.dsviewer.view_menu.AppendCheckItem(BG_SUBTRACT, 'Subtract Background')
        wx.EVT_MENU(self.dsviewer, BG_SUBTRACT, self.OnToggleBackground)
        

        #a timer object to update for us
        self.timer = mytimer()
        self.timer.Start(10000)

        self.analDispMode = 'z'

        self.numAnalysed = 0
        self.numEvents = 0

        dsviewer.paneHooks.append(self.GenPointFindingPanel)
        dsviewer.paneHooks.append(self.GenAnalysisPanel)
        dsviewer.paneHooks.append(self.GenFitStatusPanel)

        dsviewer.updateHooks.append(self.update)
        dsviewer.statusHooks.append(self.GetStatusText)

        if 'Protocol.DataStartsAt' in self.image.mdh.getEntryNames():
            self.do.zp = self.image.mdh.getEntry('Protocol.DataStartsAt')
        else:
            self.do.zp = self.image.mdh.getEntry('EstimatedLaserOnFrameNo')
        
        if (len(self.fitResults) > 0) and not 'PYME_BUGGYOPENGL' in os.environ.keys():
            self.GenResultsView()

    def GenResultsView(self):
        self.view.pointMode = 'lm'

        voxx = 1e3*self.image.mdh.getEntry('voxelsize.x')
        voxy = 1e3*self.image.mdh.getEntry('voxelsize.y')
        self.view.points = numpy.vstack((self.fitResults['fitResults']['x0']/voxx, self.fitResults['fitResults']['y0']/voxy, self.fitResults['tIndex'])).T

        if 'Splitter' in self.image.mdh.getEntry('Analysis.FitModule'):
            self.view.pointMode = 'splitter'
            self.view.pointColours = self.fitResults['fitResults']['Ag'] > self.fitResults['fitResults']['Ar']

        self.fitInf = fitInfo.FitInfoPanel(self.dsviewer, self.fitResults, self.resultsMdh, self.do.ds)
        self.dsviewer.AddPage(page=self.fitInf, select=False, caption='Fit Info')

        from PYME.Analysis.LMVis import gl_render
        self.glCanvas = gl_render.LMGLCanvas(self.dsviewer, False, vp = self.do, vpVoxSize = voxx)
        self.glCanvas.cmap = pylab.cm.gist_rainbow
        self.glCanvas.pointSelectionCallbacks.append(self.OnPointSelect)

        self.dsviewer.AddPage(page=self.glCanvas, select=True, caption='VisLite')

        xsc = self.image.data.shape[0]*1.0e3*self.image.mdh.getEntry('voxelsize.x')/self.glCanvas.Size[0]
        ysc = self.image.data.shape[1]*1.0e3*self.image.mdh.getEntry('voxelsize.y')/ self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(0, xsc*self.glCanvas.Size[0], 0, xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(0, ysc*self.glCanvas.Size[0], 0, ysc*self.glCanvas.Size[1])

        #we have to wait for the gui to be there before we start changing stuff in the GL view
        #self.timer.WantNotification.append(self.AddPointsToVis)

        self.glCanvas.Bind(wx.EVT_IDLE, self.OnIdle)
        self.pointsAdded = False
        
    def OnPointSelect(self, xp, yp):
        dist = np.sqrt((xp - self.fitResults['fitResults']['x0'])**2 + (yp - self.fitResults['fitResults']['y0'])**2)
        #print cand.sum()
        
        cand = dist.argmin()
        
        
        self.dsviewer.do.xp = xp/(1.0e3*self.image.mdh.getEntry('voxelsize.x'))
        self.dsviewer.do.yp = yp/(1.0e3*self.image.mdh.getEntry('voxelsize.y'))
        self.dsviewer.do.zp = self.fitResults['tIndex'][cand]
        

    def OnIdle(self,event):
        if not self.pointsAdded:
            self.pointsAdded = True

            self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
            self.glCanvas.setCLim((0, self.fitResults['tIndex'].max()))

    def OnToggleBackground(self, event):
        if self.do.ds.bgRange == None:
            self.do.ds.bgRange = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
            self.do.ds.dataStart = int(self.tStartAt.GetValue())
        else:
            self.do.ds.bgRange = None
            self.do.ds.dataStart = 0
            
        self.do.Optimise()

    def AddPointsToVis(self):
        self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
        self.glCanvas.setCLim((0, self.fitResults['tIndex'].max()))

        self.timer.WantNotification.remove(self.AddPointsToVis)

    def GetStatusText(self):
        return 'Frames Analysed: %d    Events detected: %d' % (self.numAnalysed, self.numEvents)

    def GenAnalysisPanel(self, _pnl):
#        item = _pnl.AddFoldPanel("Analysis", collapsed=False,
#                                      foldIcons=self.Images)
        item = afp.foldingPane(_pnl, -1, caption="Analysis", pinned = True)

        pan = wx.Panel(item, -1)

        #find out what fit factories we have
        import PYME.Analysis.FitFactories
        fitFactoryList = glob.glob(PYME.Analysis.FitFactories.__path__[0] + '/[a-zA-Z]*.py')
        fitFactoryList = [os.path.split(p)[-1][:-3] for p in fitFactoryList]
        fitFactoryList.sort()

        self.fitFactories = []
        for ff in fitFactoryList:
            try:
                fm = __import__('PYME.Analysis.FitFactories.' + ff, fromlist=['PYME', 'Analysis', 'FitFactories'])
                if 'FitResultsDType' in dir(fm):
                    self.fitFactories.append(ff)
            except:
                pass

        interpolatorList = glob.glob(PYME.Analysis.FitFactories.__path__[0] + '/Interpolators/[a-zA-Z]*.py')
        interpolatorList = [os.path.split(p)[-1][:-3] for p in interpolatorList]
        #print interpolatorList
        interpolatorList.remove('baseInterpolator')
        interpolatorList.sort()

        self.interpolators = interpolatorList

        #ditch the 'Interpolator' at the end of the module name for display
        interpolatorList = [i[:-12] for i in interpolatorList]

        #print fitFactoryList
        #print self.fitFactories

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Type:'), 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.cFitType = wx.Choice(pan, -1, choices = self.fitFactories, size=(110, -1))
        
        if 'Analysis.FitModule' in self.image.mdh.getEntryNames():
            #has already been analysed - most likely to want the same method again
            self.cFitType.SetSelection(self.fitFactories.index(self.image.mdh['Analysis.FitModule']))
            self.tThreshold.SetValue('%s' % self.image.mdh['Analysis.DetectionThreshold'])
        elif 'Camera.ROIPosY' in self.image.mdh.getEntryNames() and (self.image.mdh.getEntry('Camera.ROIHeight') + 1 + 2*(self.image.mdh.getEntry('Camera.ROIPosY')-1)) == 512:
            #we have a symetrical ROI about the centre - most likely want to analyse using splitter
            self.cFitType.SetSelection(self.fitFactories.index('SplitterFitQR'))
            self.tThreshold.SetValue('0.5')
        else:
            self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))

        hsizer.Add(self.cFitType, 1,wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Interp:'), 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.cInterpType = wx.Choice(pan, -1, choices = interpolatorList, size=(100, -1))

        self.cInterpType.SetSelection(interpolatorList.index('Linear'))

        hsizer.Add(self.cInterpType, 1,wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 7)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Start at:'), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        if 'Protocol.DataStartsAt' in self.image.mdh.getEntryNames():
            startAt = self.image.mdh.getEntry('Protocol.DataStartsAt')
        else:
            startAt = self.do.zp
        self.tStartAt = wx.TextCtrl(pan, -1, value='%d' % startAt, size=(50, -1))

        hsizer.Add(self.tStartAt, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Background:'), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBackgroundFrames = wx.TextCtrl(pan, -1, value='-30:0', size=(50, -1))

        hsizer.Add(self.tBackgroundFrames, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Debounce r:'), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tDebounceRadius = wx.TextCtrl(pan, -1, value='4', size=(50, -1))

        hsizer.Add(self.tDebounceRadius, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 10)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Z Shift [nm]:'), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tZShift = wx.TextCtrl(pan, -1, value='0', size=(50, -1))
        if 'Analysis.AxialShift' in self.image.mdh.getEntryNames():
            self.tZShift.SetValue('%3.2f' % self.image.mdh['Analysis.AxialShift'])

        hsizer.Add(self.tZShift, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 10)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        shiftFieldText = 'Shifts: <None>'
        haveShiftField=False
        if 'chroma.ShiftFilename' in self.image.mdh.getEntryNames():
            #have new format shift field data
            shiftFieldText = 'Shifts: ' + os.path.split(self.image.mdh.getEntry('chroma.ShiftFilename'))[1]
            haveShiftField=True
        elif 'chroma.dx' in self.image.mdh.getEntryNames():
            #have shift field, but filename not recorded
            shiftFieldText = 'Shifts: present'
            haveShiftField=True

        self.stShiftFieldName = wx.StaticText(pan, -1, shiftFieldText)
        if haveShiftField:
            self.stShiftFieldName.SetForegroundColour(wx.Colour(0, 128, 0))
        hsizer.Add(self.stShiftFieldName, 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        bSetShiftField = wx.Button(pan, -1, 'Set', style=wx.BU_EXACTFIT)
        bSetShiftField.Bind(wx.EVT_BUTTON, self.SetShiftField)
        hsizer.Add(bSetShiftField, 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 0)

        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        psfFieldText = 'PSF: <None>'
        havePSF = False

        if 'PSFFile' in self.image.mdh.getEntryNames():
            psfFieldText = 'PSF: ' + os.path.split(self.image.mdh.getEntry('PSFFile'))[1]
            havePSF = True

        self.stPSFFilename = wx.StaticText(pan, -1, psfFieldText)
        if havePSF:
            self.stPSFFilename.SetForegroundColour(wx.Colour(0, 128, 0))

        hsizer.Add(self.stPSFFilename, 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        bSetPSF = wx.Button(pan, -1, 'Set', style=wx.BU_EXACTFIT)
        bSetPSF.Bind(wx.EVT_BUTTON, self.SetPSF)
        hsizer.Add(bSetPSF, 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 0)

        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 10)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, 5, 5)
        item.AddNewElement(pan)

        self.cbDrift = wx.CheckBox(item, -1, 'Estimate Drift')
        self.cbDrift.SetValue(False)

        #_pnl.AddFoldPanelWindow(item, self.cbDrift, fpb.FPB_ALIGN_WIDTH, 7, 5)
        item.AddNewElement(self.cbDrift)

        self.cbSubtractBackground = wx.CheckBox(item, -1, 'Subtract background in fit')
        self.cbSubtractBackground.SetValue(True)

        #_pnl.AddFoldPanelWindow(item, self.cbSubtractBackground, fpb.FPB_ALIGN_WIDTH, 2, 5)
        item.AddNewElement(self.cbSubtractBackground)

        self.bGo = wx.Button(item, -1, 'Go')


        self.bGo.Bind(wx.EVT_BUTTON, self.OnGo)
        #_pnl.AddFoldPanelWindow(item, self.bGo, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(self.bGo)
        _pnl.AddPane(item)

        self.analysisPanel = item


    def SetPSF(self, event=None):
        fdialog = wx.FileDialog(None, 'Please select PSF to use ...',
                    defaultDir=os.path.split(self.image.filename)[0],
                    wildcard='PSF files|*.psf', style=wx.OPEN)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            #self.ds = example.CDataStack(fdialog.GetPath().encode())
            #self.ds =
            psfFilename = fdialog.GetPath()
            self.image.mdh.setEntry('PSFFile', getRelFilename(psfFilename))
            #self.md.setEntry('PSFFile', psfFilename)
            self.stPSFFilename.SetLabel('PSF: %s' % os.path.split(psfFilename)[1])
            self.stPSFFilename.SetForegroundColour(wx.Colour(0, 128, 0))
            return True
        else:
            return False

    def SetShiftField(self, event=None):
        fdialog = wx.FileDialog(None, 'Please select shift field to use ...',
                    wildcard='Shift fields|*.sf', style=wx.OPEN, defaultDir = nameUtils.genShiftFieldDirectoryPath())
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            #self.ds = example.CDataStack(fdialog.GetPath().encode())
            #self.ds =
            sfFilename = fdialog.GetPath()
            self.image.mdh.setEntry('chroma.ShiftFilename', sfFilename)
            dx, dy = numpy.load(sfFilename)
            self.image.mdh.setEntry('chroma.dx', dx)
            self.image.mdh.setEntry('chroma.dy', dy)
            #self.md.setEntry('PSFFile', psfFilename)
            self.stShiftFieldName.SetLabel('Shifts: %s' % os.path.split(sfFilename)[1])
            self.stShiftFieldName.SetForegroundColour(wx.Colour(0, 128, 0))
            return True
        else:
            return False


    def OnGo(self, event):
        threshold = float(self.tThreshold.GetValue())
        startAt = int(self.tStartAt.GetValue())
        driftEst = self.cbDrift.GetValue()
        fitMod = self.cFitType.GetStringSelection()
        interpolator = self.interpolators[self.cInterpType.GetSelection()]
        bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]

        self.image.mdh.setEntry('Analysis.subtractBackground', self.cbSubtractBackground.GetValue())
        #self.md.setEntry('Analysis.subtractBackground', self.cbSubtractBackground.GetValue())

        #self.image.mdh.setEntry('Analysis.NumBGFrames', bgFrames)
        self.image.mdh.setEntry('Analysis.BGRange', bgFrames)
        #self.md.setEntry('Analysis.NumBGFrames', bgFrames)

        self.image.mdh.setEntry('Analysis.InterpModule', interpolator)

        self.image.mdh.setEntry('Analysis.DebounceRadius', int(self.tDebounceRadius.GetValue()))

        if fitMod.startswith('SplitterFit') and not 'chroma.dx' in self.image.mdh.getEntryNames():
            if not self.SetShiftField():
                return

        if 'Interp' in fitMod and not 'PSFFile' in self.image.mdh.getEntryNames():
            if not self.SetPSF():
                return

        if 'Interp' in fitMod  and 'Splitter' in fitMod:
            #dlg = wx.TextEntryDialog(self, 'What is the axial chromatic shift between splitter halves [nm]?',
            #    'Axial Shift', '300')

            #if dlg.ShowModal() == wx.ID_OK:
            #    self.image.mdh.setEntry('Analysis.AxialShift', float(dlg.GetValue()))
            #else:
            self.image.mdh.setEntry('Analysis.AxialShift', float(self.tZShift.GetValue()))


            #dlg.Destroy()
            
        if debug:
            print 'About to push images'

        if not driftEst:
            self.pushImages(startAt, threshold, fitMod)
        else:
            self.pushImagesD(startAt, threshold)
            
        if debug:
            print 'Images pushed'

        from PYME.Analysis.LMVis import gl_render
        self.glCanvas = gl_render.LMGLCanvas(self.dsviewer, False)
        self.glCanvas.cmap = pylab.cm.gist_rainbow

        self.dsviewer.AddPage(page=self.glCanvas, select=True, caption='VisLite')
        

        xsc = self.image.data.shape[0]*1.0e3*self.image.mdh.getEntry('voxelsize.x')/self.glCanvas.Size[0]
        ysc = self.image.data.shape[1]*1.0e3*self.image.mdh.getEntry('voxelsize.y')/ self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(0, xsc*self.glCanvas.Size[0], 0, xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(0, ysc*self.glCanvas.Size[0], 0, ysc*self.glCanvas.Size[1])

        self.numAnalysed = 0
        self.numEvents = 0
        self.fitResults = []

        self.timer.WantNotification.append(self.analRefresh)
        self.bGo.Enable(False)
        #_pnl.Collapse(self.analysisPanel)

    def GenPointFindingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Point Finding", pinned = True)
#        item = _pnl.AddFoldPanel("Point Finding", collapsed=False,
#                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Thresh:'), 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tThreshold = wx.TextCtrl(pan, -1, value='0.6', size=(30, -1))

        hsizer.Add(self.tThreshold, 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        bTest = wx.Button(pan, -1, 'Test', style=wx.BU_EXACTFIT)
        bTest.Bind(wx.EVT_BUTTON, self.OnTest)
        hsizer.Add(bTest, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        
        bTestF = wx.Button(pan, -1, 'TF', style=wx.BU_EXACTFIT)
        bTestF.Bind(wx.EVT_BUTTON, self.OnTestFrame)
        hsizer.Add(bTestF, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        item.AddNewElement(pan)
        _pnl.AddPane(item)


        #_pnl.AddFoldPanelWindow(item, bTest, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

    def OnTest(self, event):
        threshold = float(self.tThreshold.GetValue())
        startAt = int(self.tStartAt.GetValue())
        driftEst = self.cbDrift.GetValue()
        fitMod = self.cFitType.GetStringSelection()

        self.image.mdh.setEntry('Analysis.DebounceRadius', int(self.tDebounceRadius.GetValue()))

        if 'Psf' in fitMod and not 'PSFFile' in self.image.mdh.getEntryNames():
            fdialog = wx.FileDialog(None, 'Please select PSF to use ...',
                    wildcard='PSF files|*.psf', style=wx.OPEN)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                #self.ds = example.CDataStack(fdialog.GetPath().encode())
                #self.ds =
                psfFilename = fdialog.GetPath()
                self.image.mdh.setEntry('PSFFile', getRelFilename(psfFilename))
                #self.md.setEntry('PSFFile', psfFilename)
            else:
                return

        #if not driftEst:
        self.testFrames(threshold)
        
    def OnTestFrame(self, event):
        threshold = float(self.tThreshold.GetValue())
        startAt = int(self.tStartAt.GetValue())
        driftEst = self.cbDrift.GetValue()
        fitMod = self.cFitType.GetStringSelection()

        self.image.mdh.setEntry('Analysis.DebounceRadius', int(self.tDebounceRadius.GetValue()))

        if 'Psf' in fitMod and not 'PSFFile' in self.image.mdh.getEntryNames():
            fdialog = wx.FileDialog(None, 'Please select PSF to use ...',
                    wildcard='PSF files|*.psf', style=wx.OPEN)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                #self.ds = example.CDataStack(fdialog.GetPath().encode())
                #self.ds =
                psfFilename = fdialog.GetPath()
                self.image.mdh.setEntry('PSFFile', getRelFilename(psfFilename))
                #self.md.setEntry('PSFFile', psfFilename)
            else:
                return

        #if not driftEst:
        self.testFrame(threshold)
        #else:
        #    self.sh.run('pushImagesD(%d, %f)' % (startAt, threshold)

    def GenFitStatusPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Fit Status", pinned = True)
#        item = _pnl.AddFoldPanel("Fit Status", collapsed=False,
#                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1, size = (160, 300))

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Colour:'), 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chProgDispColour = wx.Choice(pan, -1, choices = ['z', 'gFrac', 't'], size=(60, -1))
        self.chProgDispColour.Bind(wx.EVT_CHOICE, self.OnProgDispColourChange)
        hsizer.Add(self.chProgDispColour, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)

        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'CMap:'), 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chProgDispCMap = wx.Choice(pan, -1, choices = ['gist_rainbow', 'RdYlGn'], size=(60, -1))
        self.chProgDispCMap.Bind(wx.EVT_CHOICE, self.OnProgDispCMapChange)
        hsizer.Add(self.chProgDispCMap, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)

        vsizer.Add(hsizer, 0,wx.BOTTOM|wx.EXPAND, 7)

        self.progPan = progGraph.progPanel(pan, self.fitResults, size=(220, 250))
        self.progPan.draw()

        vsizer.Add(self.progPan, 1,wx.ALL|wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 0)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 0)
        item.AddNewElement(pan)
        _pnl.AddPane(item)

    def OnProgDispColourChange(self, event):
        #print 'foo'
        self.analDispMode = self.chProgDispColour.GetStringSelection()
        self.analRefresh()

    def OnProgDispCMapChange(self, event):
        #print 'foo'
        self.glCanvas.setCMap(pylab.cm.__getattribute__(self.chProgDispCMap.GetStringSelection()))

    def OnCalibrateSplitter(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('SplitterShiftEstFR'))
        self.tBackgroundFrames.SetValue('0:0')
        self.cbSubtractBackground.SetValue(False)
        self.tThreshold.SetValue('2')
        self.image.mdh['Analysis.ChunkSize'] = 1
        self.image.mdh['Analysis.ROISize'] = 11

    def OnStandard2D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('0.6')
        
    def OnSpitter2D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('SplitterFitQR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('0.5')
    
    def OnStandard3D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('InterpFitR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('1.0')
    
    def OnSpliter3D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('SplitterFitInterpR'))
        self.tBackgroundFrames.SetValue('-30:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('1.0')
        
    def OnPRI3D(self, event):
        #self.cFitType.SetSelection(self.fitFactories.index('SplitterFitInterpR'))
        #self.tBackgroundFrames.SetValue('-30:0')
        #self.cbSubtractBackground.SetValue(True)
        #self.tThreshold.SetValue('1.0')
        self.image.mdh['PRI.Axis'] = 'y'
        self.image.mdh['Analysis.EstimatorModule'] = 'priEstimator'

    def analRefresh(self):
        newNumAnalysed = self.tq.getNumberTasksCompleted(self.image.seriesName)
        if newNumAnalysed > self.numAnalysed:
            self.numAnalysed = newNumAnalysed
            newResults = self.tq.getQueueData(self.image.seriesName, 'FitResults', len(self.fitResults))
            if len(newResults) > 0:
                if len(self.fitResults) == 0:
                    self.fitResults = newResults
                else:
                    self.fitResults = numpy.concatenate((self.fitResults, newResults))
                self.progPan.fitResults = self.fitResults

                self.view.points = numpy.vstack((self.fitResults['fitResults']['x0'], self.fitResults['fitResults']['y0'], self.fitResults['tIndex'])).T

                self.numEvents = len(self.fitResults)

                if self.analDispMode == 'z' and (('zm' in dir(self)) or ('z0' in self.fitResults['fitResults'].dtype.fields)):
                    #display z as colour
                    if 'zm' in dir(self): #we have z info
                        if 'z0' in self.fitResults['fitResults'].dtype.fields:
                            z = 1e3*self.zm(self.fitResults['tIndex'].astype('f')).astype('f')
                            z_min = z.min() - 500
                            z_max = z.max() + 500
                            z = z + self.fitResults['fitResults']['z0']
                            self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],z)
                            self.glCanvas.setCLim((z_min, z_max))
                        else:
                            z = self.zm(self.fitResults['tIndex'].astype('f')).astype('f')
                            self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],z)
                            self.glCanvas.setCLim((z.min(), z.max()))
                    elif 'z0' in self.fitResults['fitResults'].dtype.fields:
                        z = self.fitResults['fitResults']['z0']
                        self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],z)
                        self.glCanvas.setCLim((-1e3, 1e3))

                elif self.analDispMode == 'gFrac' and 'Ag' in self.fitResults['fitResults'].dtype.fields:
                    #display ratio of colour channels as point colour
                    c = self.fitResults['fitResults']['Ag']/(self.fitResults['fitResults']['Ag'] + self.fitResults['fitResults']['Ar'])
                    self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],c)
                    self.glCanvas.setCLim((0, 1))
                elif self.analDispMode == 'gFrac' and 'ratio' in self.fitResults['fitResults'].dtype.fields:
                    #display ratio of colour channels as point colour
                    c = self.fitResults['fitResults']['ratio']
                    self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],c)
                    self.glCanvas.setCLim((0, 1))
                else:
                    #default to time
                    self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
                    self.glCanvas.setCLim((0, self.numAnalysed))

        if (self.tq.getNumberOpenTasks(self.image.seriesName) + self.tq.getNumberTasksInProgress(self.image.seriesName)) == 0 and 'SpoolingFinished' in self.image.mdh.getEntryNames():
            self.dsviewer.statusbar.SetBackgroundColour(wx.GREEN)
            self.dsviewer.statusbar.Refresh()

        self.progPan.draw()
        self.progPan.Refresh()
        self.dsviewer.Refresh()
        self.dsviewer.update()

    def update(self, dsviewer):
        if 'fitInf' in dir(self) and not self.dsviewer.playbackpanel.tPlay.IsRunning():
            self.fitInf.UpdateDisp(self.view.PointsHitTest())


    #from fth5.py
    def checkTQ(self):
        if self.tq == None:
            #if 'PYME_TASKQUEUENAME' in os.environ.keys():
            #    taskQueueName = os.environ['PYME_TASKQUEUENAME']
            #else:
            #    taskQueueName = 'taskQueue'

            from PYME.misc.computerName import GetComputerName
            compName = GetComputerName()
            
            try:

                taskQueueName = 'TaskQueues.%s' % compName

                self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)
            except:
                taskQueueName = 'PrivateTaskQueues.%s' % compName

                self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)


    def pushImages(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        self.checkTQ()
        if debug:
            print 'TQ checked'
        if self.image.dataSource.moduleName == 'HDFDataSource':
            self.pushImagesHDF(startingAt, detThresh, fitFcn)
        elif self.image.dataSource.moduleName == 'TQDataSource':
            self.pushImagesQueue(startingAt, detThresh, fitFcn)
        else: #generic catchall for other data sources
            self.pushImagesDS(startingAt, detThresh, fitFcn)


    def pushImagesHDF(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        #global seriesName
        dataFilename = self.image.seriesName
        resultsFilename = genResultFileName(self.image.seriesName)

        if os.path.exists(resultsFilename):
            di, fn = os.path.split(resultsFilename)
            i = 1
            stub = os.path.splitext(fn)[0]
            while os.path.exists(os.path.join(di, stub + '_%d.h5r' % i)):
                i += 1
            fdialog = wx.FileDialog(None, 'Analysis file already exists, please select a new filename',
                        wildcard='H5R files|*.h5r', defaultDir=di, defaultFile=stub + '_%d.h5r' % i, style=wx.SAVE)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                resultsFilename = fdialog.GetPath().encode()
            else:
                raise RuntimeError('Invalid results file - not running')
            self.image.seriesName = resultsFilename

        self.tq.createQueue('HDFTaskQueue', self.image.seriesName, dataFilename = dataFilename, resultsFilename=resultsFilename, startAt = 'notYet')

        mdhQ = MetaDataHandler.QueueMDHandler(self.tq, self.image.seriesName, self.image.mdh)
        mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
        mdhQ.setEntry('Analysis.FitModule', fitFcn)
        mdhQ.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.image.dataSource))

        evts = self.image.dataSource.getEvents()
        if len(evts) > 0:
            self.tq.addQueueEvents(self.image.seriesName, evts)

        self.tq.releaseTasks(self.image.seriesName, startingAt)


    def pushImagesQueue(self, startingAt=0, detThresh = .9, fitFcn='LatGaussFitFR'):
        self.image.mdh.setEntry('Analysis.DetectionThreshold', detThresh)
        self.image.mdh.setEntry('Analysis.FitModule', fitFcn)
        self.image.mdh.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.image.dataSource))
        self.tq.releaseTasks(self.image.seriesName, startingAt)

    def pushImagesDS(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        #global seriesName
        #dataFilename = self.image.seriesName
        resultsFilename = genResultFileName(self.image.seriesName)
        while os.path.exists(resultsFilename):
            di, fn = os.path.split(resultsFilename)
            fdialog = wx.FileDialog(None, 'Analysis file already exists, please select a new filename',
                        wildcard='H5R files|*.h5r', defaultDir=di, defaultFile=os.path.splitext(fn)[0] + '_1.h5r', style=wx.SAVE)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                resultsFilename = fdialog.GetPath().encode()
            else:
                raise RuntimeError('Invalid results file - not running')
            #self.image.seriesName = resultsFilename
            
        debugPrint('Results file = %s' % resultsFilename) 

        self.tq.createQueue('HDFResultsTaskQueue', resultsFilename, None)
        
        debugPrint('Queue created')

        mdhQ = MetaDataHandler.QueueMDHandler(self.tq, resultsFilename, self.image.mdh)
        mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
        mdhQ.setEntry('Analysis.FitModule', fitFcn)
        mdhQ.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.image.dataSource))
        
        debugPrint('Metadata transferred to queue')

        evts = self.image.dataSource.getEvents()
        if len(evts) > 0:
            self.tq.addQueueEvents(resultsFilename, evts)

        md = MetaDataHandler.NestedClassMDHandler(mdhQ)

        mn = self.image.dataSource.moduleName

        #if it's a buffered source, go back to underlying source
        if mn == 'BufferedDataSource':
            mn = self.image.dataSource.dataSource.moduleName

        for i in range(startingAt, self.image.dataSource.getNumSlices()):
            debugPrint('Posting task %d' %i)
            if 'Analysis.BGRange' in md.getEntryNames():
                bgi = range(max(i + md.Analysis.BGRange[0],md.EstimatedLaserOnFrameNo), max(i + md.Analysis.BGRange[1],md.EstimatedLaserOnFrameNo))
            elif 'Analysis.NumBGFrames' in md.getEntryNames():
                bgi = range(max(i - md.Analysis.NumBGFrames, md.EstimatedLaserOnFrameNo), i)
            else:
                bgi = range(max(i - 10, md.EstimatedLaserOnFrameNo), i)

            #task = fitTask(self.queueID, taskNum, self.metaData.Analysis.DetectionThreshold, self.metaData, self.metaData.Analysis.FitModule, 'TQDataSource', bgindices =bgi, SNThreshold = True)
            
            self.tq.postTask(remFitBuf.fitTask(self.image.seriesName,i, detThresh, md, fitFcn, bgindices=bgi, SNThreshold=True, dataSourceModule=mn), queueName=resultsFilename)

        self.image.seriesName = resultsFilename
            
        #self.tq.releaseTasks(self.image.seriesName, startingAt)


#    def testFrame(self, detThresh = 0.9):
#        ft = remFitBuf.fitTask(self.image.seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), cFitType.GetString(cFitType.GetSelection()), bgindices=range(max(vp.zp-10, self.image.mdh.getEntry('EstimatedLaserOnFrameNo')),vp.zp), SNThreshold=True)
#        return ft(True)
#
#    def testFrameTQ(self, detThresh = 0.9):
#        ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatGaussFitFR', 'TQDataSource', bgindices=range(max(vp.zp-10, mdh.getEntry('EstimatedLaserOnFrameNo')),vp.zp), SNThreshold=True)
#        return ft(True, tq)
#
#    def pushImagesD(self, startingAt=0, detThresh = .9):
#        self.tq.createQueue('HDFResultsTaskQueue', self.image.seriesName, None)
#        mdhQ = MetaDataHandler.QueueMDHandler(self.tq, self.image.seriesName, mdh)
#        mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
#        for i in range(startingAt, ds.shape[0]):
#            self.tq.postTask(remFitBuf.fitTask(self.image.seriesName,i, detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'LatGaussFitFR', bgindices=range(max(i-10,self.image.mdh.getEntry('EstimatedLaserOnFrameNo') ),i), SNThreshold=True,driftEstInd=range(max(i-5, self.image.mdh.getEntry('EstimatedLaserOnFrameNo')),min(i + 5, ds.shape[0])), dataSourceModule=self.dataSource.moduleName), queueName=self.image.seriesName)


#    def testFrameD(self, detThresh = 0.9):
#        ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True,driftEstInd=range(max(vp.zp-5, md.EstimatedLaserOnFrameNo),min(vp.zp + 5, ds.shape[0])))
#        return ft(True)

    def testFrames(self, detThresh = 0.9, offset = 0):
        close('all')
        if self.image.dataSource.moduleName == 'TQDataSource':
            self.checkTQ()
        matplotlib.interactive(False)
        clf()

        sq = min(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 1000, self.image.dataSource.getNumSlices()/4)
        zps = array(range(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 20, self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 24)  + range(sq, sq + 4) + range(self.image.dataSource.getNumSlices()/2,self.image.dataSource.getNumSlices() /2+4))
        zps += offset
        fitMod = self.cFitType.GetStringSelection()
        #bgFrames = int(tBackgroundFrames.GetValue())
        bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
        print zps
        for i in range(12):
            print i
            #if 'Analysis.NumBGFrames' in md.getEntryNames():
            #bgi = range(max(zps[i] - bgFrames,mdh.getEntry('EstimatedLaserOnFrameNo')), zps[i])
            bgi = range(max(zps[i] + bgFrames[0],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')), max(zps[i] + bgFrames[1],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')))
            #else:
            #    bgi = range(max(zps[i] - 10,md.EstimatedLaserOnFrameNo), zps[i])
            mn = self.image.dataSource.moduleName
            if mn == 'BufferedDataSource':
                mn = self.image.dataSource.dataSource.moduleName

            if 'Splitter' in fitMod:
                ft = remFitBuf.fitTask(self.image.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'SplitterObjFindR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
            else:
                ft = remFitBuf.fitTask(self.image.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'LatObjFindFR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
            res = ft(taskQueue=self.tq)
            xp = floor(i/4)/3.
            yp = (3 - i%4)/4.
            #print xp, yp
            axes((xp,yp, 1./6,1./4.5))
            #d = ds[zps[i], :,:].squeeze().T
            d = self.image.dataSource.getSlice(zps[i]).T
            imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
            title('Frame %d' % zps[i])
            xlim(0, d.shape[1])
            ylim(0, d.shape[0])
            xticks([])
            yticks([])
            #print 'i = %d, ft.index = %d' % (i, ft.index)
            #subplot(4,6,2*i+13)
            xp += 1./6
            axes((xp,yp, 1./6,1./4.5))
            d = ft.ofd.filteredData.T
            #d = ft.data.squeeze().T
            imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
            plot([p.x for p in ft.ofd], [p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
            if ft.driftEst:
                 plot([p.x for p in ft.ofdDr], [p.y for p in ft.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
            if ft.fitModule in remFitBuf.splitterFitModules:
                if self.image.mdh.getEntry('Splitter.Flip'):
                    plot([p.x for p in ft.ofd], [ d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
                else:
                    plot([p.x for p in ft.ofd], [ p.y + d.shape[0]/2 for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
            axis('tight')
            xlim(0, d.shape[1])
            ylim(0, d.shape[0])
            xticks([])
            yticks([])
        show()
        matplotlib.interactive(True)
        
    def testFrame(self, detThresh = 0.9, offset = 0, gui=True):
        #close('all')
        if self.image.dataSource.moduleName == 'TQDataSource':
            self.checkTQ()
        if gui:    
            matplotlib.interactive(False)
            figure()
        #sq = min(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 1000, self.image.dataSource.getNumSlices()/4)
        #zps = array(range(self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 20, self.image.mdh.getEntry('EstimatedLaserOnFrameNo') + 24)  + range(sq, sq + 4) + range(self.image.dataSource.getNumSlices()/2,self.image.dataSource.getNumSlices() /2+4))
        #zps += offset
        
        zp = self.do.zp
        fitMod = self.cFitType.GetStringSelection()
        #bgFrames = int(tBackgroundFrames.GetValue())
        bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
        #print zps
        bgi = range(max(zp + bgFrames[0],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')), max(zp + bgFrames[1],self.image.mdh.getEntry('EstimatedLaserOnFrameNo')))
            #else:
            #    bgi = range(max(zps[i] - 10,md.EstimatedLaserOnFrameNo), zps[i])
        mn = self.image.dataSource.moduleName
        if mn == 'BufferedDataSource':
            mn = self.image.dataSource.dataSource.moduleName

        #if 'Splitter' in fitMod:
        #    ft = remFitBuf.fitTask(self.image.seriesName, zp, detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'SplitterObjFindR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
        #else:
        #    ft = remFitBuf.fitTask(self.image.seriesName, zp, detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), 'LatObjFindFR', bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
        ft = remFitBuf.fitTask(self.image.seriesName, zp, detThresh, MetaDataHandler.NestedClassMDHandler(self.image.mdh), fitMod, bgindices=bgi, SNThreshold=True,dataSourceModule=mn)
        res = ft(gui=gui,taskQueue=self.tq)
        
        if gui:
            figure()
            try:
                d = ft.ofd.filteredData.T
                #d = ft.data.squeeze().T
                imshow(d, cmap=cm.hot, interpolation='nearest', hold=False, clim=(median(d.ravel()), d.max()))
                plot([p.x for p in ft.ofd], [p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
                if ft.driftEst:
                     plot([p.x for p in ft.ofdDr], [p.y for p in ft.ofdDr], 'o', mew=2, mec='b', mfc='none', ms=9)
                #if ft.fitModule in remFitBuf.splitterFitModules:
                #        plot([p.x for p in ft.ofd], [d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
                #axis('tight')
                xlim(0, d.shape[1])
                ylim(d.shape[0], 0)
                xticks([])
                yticks([])
                
                if 'tIm' in dir(ft.ofd):
                    figure()
                    imshow(ft.ofd.tIm.T, cmap=cm.hot, interpolation='nearest', hold=False)
                    #axis('tight')
                    xlim(0, d.shape[1])
                    ylim(d.shape[0], 0)
                    xticks([])
                    yticks([])
                    
                vx = 1e3*self.image.mdh['voxelsize.x']
                vy = 1e3*self.image.mdh['voxelsize.y']
                plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, '+b')
                    
                #figure()
                #imshow()
            except AttributeError:
                #d = self.image.data[:,:,zp].squeeze().T
                d = (ft.data.squeeze() - ft.bg.squeeze()).T
                imshow(d, cmap=cm.jet, interpolation='nearest', clim = [0, d.max()])
                xlim(0, d.shape[1])
                ylim(d.shape[0], 0)
                
                vx = 1e3*self.image.mdh['voxelsize.x']
                vy = 1e3*self.image.mdh['voxelsize.y']
                plot(res.results['fitResults']['x0']/vx, res.results['fitResults']['y0']/vy, 'ow')
                pass
                    
            show()
    
            matplotlib.interactive(True)
        
        return ft, res

def Plug(dsviewer):
    dsviewer.LMAnalyser = LMAnalyser(dsviewer)

    if not 'overlaypanel' in dir(dsviewer):    
        dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
        dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
        pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)
    
        dsviewer.panesToMinimise.append(pinfo2)

