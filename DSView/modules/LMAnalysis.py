#!/usr/bin/python
##################
# LMAnalysis.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
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

class LMAnalyser:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        if 'tq' in dir(dsviewer):
            self.tq = dsviewer.tq
        else:
            self.tq = None

        self.dataSource = dsviewer.dataSource
        self.ds = dsviewer.ds
        self.mdh = dsviewer.mdh
        self.seriesName = dsviewer.seriesName
        self.vp = dsviewer.vp
        self.fitResults = dsviewer.fitResults
        self.resultsMdh = dsviewer.resultsMdh

        mTasks = wx.Menu()
        TASKS_STANDARD_2D = wx.NewId()
        TASKS_CALIBRATE_SPLITTER = wx.NewId()
        mTasks.Append(TASKS_STANDARD_2D, "Normal 2D analysis", "", wx.ITEM_NORMAL)
        mTasks.Append(TASKS_CALIBRATE_SPLITTER, "Calibrating the splitter", "", wx.ITEM_NORMAL)
        self.dsviewer.menubar.Append(mTasks, "Set defaults for")
        
        wx.EVT_MENU(self.dsviewer, TASKS_CALIBRATE_SPLITTER, self.OnCalibrateSplitter)
        wx.EVT_MENU(self.dsviewer, TASKS_STANDARD_2D, self.OnStandard2D)

        #a timer object to update for us
        self.timer = mytimer()
        self.timer.Start(10000)

        self.analDispMode = 'z'

        self.numAnalysed = 0
        self.numEvents = 0

        dsviewer.paneHooks.append(self.GenPointFindingPanel)
        dsviewer.paneHooks.append(self.GenAnalysisPanel)
        dsviewer.paneHooks.append(self.GenFitStatusPanel)
        
        if len(self.fitResults) > 0:
            self.GenResultsView()

    def GenResultsView(self):
        self.vp.view.pointMode = 'lm'

        voxx = 1e3*self.mdh.getEntry('voxelsize.x')
        voxy = 1e3*self.mdh.getEntry('voxelsize.y')
        self.vp.view.points = numpy.vstack((self.fitResults['fitResults']['x0']/voxx, self.fitResults['fitResults']['y0']/voxy, self.fitResults['tIndex'])).T

        if 'Splitter' in self.mdh.getEntry('Analysis.FitModule'):
            self.vp.view.pointMode = 'splitter'
            self.vp.view.pointColours = self.fitResults['fitResults']['Ag'] > self.fitResults['fitResults']['Ar']

        from PYME.Analysis.LMVis import gl_render
        self.glCanvas = gl_render.LMGLCanvas(self.dsviewer.notebook1, False, vp = self.vp.do, vpVoxSize = voxx)
        self.glCanvas.cmap = pylab.cm.gist_rainbow

        self.dsviewer.notebook1.AddPage(page=self.glCanvas, select=True, caption='VisLite')

        xsc = self.ds.shape[0]*1.0e3*self.mdh.getEntry('voxelsize.x')/self.glCanvas.Size[0]
        ysc = self.ds.shape[1]*1.0e3*self.mdh.getEntry('voxelsize.y')/ self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(0, xsc*self.glCanvas.Size[0], 0, xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(0, ysc*self.glCanvas.Size[0], 0, ysc*self.glCanvas.Size[1])

        #we have to wait for the gui to be there before we start changing stuff in the GL view
        self.timer.WantNotification.append(self.AddPointsToVis)

        self.fitInf = fitInfo.FitInfoPanel(self.dsviewer, self.fitResults, self.resultsMdh, self.vp.do.ds)
        self.dsviewer.notebook1.AddPage(page=self.fitInf, select=False, caption='Fit Info')

    def AddPointsToVis(self):
        self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
        self.glCanvas.setCLim((0, self.fitResults['tIndex'].max()))

        self.timer.WantNotification.remove(self.AddPointsToVis)

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

        if 'Camera.ROIPosY' in self.mdh.getEntryNames() and (self.mdh.getEntry('Camera.ROIHeight') + 1 + 2*(self.mdh.getEntry('Camera.ROIPosY')-1)) == 512:
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
        self.tStartAt = wx.TextCtrl(pan, -1, value='%d' % self.vp.do.zp, size=(50, -1))

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
        shiftFieldText = 'Shifts: <None>'
        haveShiftField=False
        if 'chroma.ShiftFilename' in self.mdh.getEntryNames():
            #have new format shift field data
            shiftFieldText = 'Shifts: ' + os.path.split(self.mdh.getEntry('chroma.ShiftFilename'))[1]
            haveShiftField=True
        elif 'chroma.dx' in self.mdh.getEntryNames():
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

        if 'PSFFile' in self.mdh.getEntryNames():
            psfFieldText = 'PSF: ' + os.path.split(self.mdh.getEntry('PSFFile'))[1]
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
                    wildcard='PSF files|*.psf', style=wx.OPEN)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            #self.ds = example.CDataStack(fdialog.GetPath().encode())
            #self.ds =
            psfFilename = fdialog.GetPath()
            self.mdh.setEntry('PSFFile', getRelFilename(psfFilename))
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
            self.mdh.setEntry('chroma.ShiftFilename', sfFilename)
            dx, dy = numpy.load(sfFilename)
            self.mdh.setEntry('chroma.dx', dx)
            self.mdh.setEntry('chroma.dy', dy)
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

        self.mdh.setEntry('Analysis.subtractBackground', self.cbSubtractBackground.GetValue())
        #self.md.setEntry('Analysis.subtractBackground', self.cbSubtractBackground.GetValue())

        #self.mdh.setEntry('Analysis.NumBGFrames', bgFrames)
        self.mdh.setEntry('Analysis.BGRange', bgFrames)
        #self.md.setEntry('Analysis.NumBGFrames', bgFrames)

        self.mdh.setEntry('Analysis.InterpModule', interpolator)

        self.mdh.setEntry('Analysis.DebounceRadius', int(self.tDebounceRadius.GetValue()))

        if fitMod.startswith('SplitterFit') and not 'chroma.dx' in self.mdh.getEntryNames():
            if not self.SetShiftField():
                return

        if 'Psf' in fitMod and not 'PSFFile' in self.mdh.getEntryNames():
            if not self.SetPSF():
                return

        if 'Psf' in fitMod  and 'Splitter' in fitMod and not 'Analysis.AxialShift' in self.mdh.getEntryNames():
            dlg = wx.TextEntryDialog(self, 'What is the axial chromatic shift between splitter halves [nm]?',
                'Axial Shift', '300')

            if dlg.ShowModal() == wx.ID_OK:
                self.mdh.setEntry('Analysis.AxialShift', float(dlg.GetValue()))
            else:
                self.mdh.setEntry('Analysis.AxialShift', 0.)


            dlg.Destroy()

        if not driftEst:
            self.sh.run('pushImages(%d, %f, "%s")' % (startAt, threshold, fitMod))
        else:
            self.sh.run('pushImagesD(%d, %f)' % (startAt, threshold))

        from PYME.Analysis.LMVis import gl_render
        self.glCanvas = gl_render.LMGLCanvas(self.notebook1, False)
        self.glCanvas.cmap = pylab.cm.gist_rainbow

        self.notebook1.AddPage(page=self.glCanvas, select=True, caption='VisLite')

        xsc = self.ds.shape[0]*1.0e3*self.mdh.getEntry('voxelsize.x')/self.glCanvas.Size[0]
        ysc = self.ds.shape[1]*1.0e3*self.mdh.getEntry('voxelsize.y')/ self.glCanvas.Size[1]

        if xsc > ysc:
            self.glCanvas.setView(0, xsc*self.glCanvas.Size[0], 0, xsc*self.glCanvas.Size[1])
        else:
            self.glCanvas.setView(0, ysc*self.glCanvas.Size[0], 0, ysc*self.glCanvas.Size[1])

        self.timer.WantNotification.append(self.analRefresh)
        self.bGo.Enable(False)
        _pnl.Collapse(self.analysisPanel)

    def GenPointFindingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Point Finding", pinned = True)
#        item = _pnl.AddFoldPanel("Point Finding", collapsed=False,
#                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tThreshold = wx.TextCtrl(pan, -1, value='0.6', size=(40, -1))

        hsizer.Add(self.tThreshold, 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        bTest = wx.Button(pan, -1, 'Test', style=wx.BU_EXACTFIT)
        bTest.Bind(wx.EVT_BUTTON, self.OnTest)
        hsizer.Add(bTest, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)

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

        self.mdh.setEntry('Analysis.DebounceRadius', int(self.tDebounceRadius.GetValue()))

        if 'Psf' in fitMod and not 'PSFFile' in self.mdh.getEntryNames():
            fdialog = wx.FileDialog(None, 'Please select PSF to use ...',
                    wildcard='PSF files|*.psf', style=wx.OPEN)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                #self.ds = example.CDataStack(fdialog.GetPath().encode())
                #self.ds =
                psfFilename = fdialog.GetPath()
                self.mdh.setEntry('PSFFile', getRelFilename(psfFilename))
                #self.md.setEntry('PSFFile', psfFilename)
            else:
                return

        #if not driftEst:
        self.sh.run('testFrames(%f)' % (threshold))
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

        self.progPan = progGraph.progPanel(pan, self.dsviewer.fitResults, size=(220, 250))
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

    def OnStandard2D(self, event):
        self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
        self.tBackgroundFrames.SetValue('-10:0')
        self.cbSubtractBackground.SetValue(True)
        self.tThreshold.SetValue('0.6')

    def analRefresh(self):
        newNumAnalysed = self.tq.getNumberTasksCompleted(self.seriesName)
        if newNumAnalysed > self.numAnalysed:
            self.numAnalysed = newNumAnalysed
            newResults = self.tq.getQueueData(self.seriesName, 'FitResults', len(self.fitResults))
            if len(newResults) > 0:
                if len(self.fitResults) == 0:
                    self.fitResults = newResults
                else:
                    self.fitResults = numpy.concatenate((self.fitResults, newResults))
                self.progPan.fitResults = self.fitResults

                self.vp.points = numpy.vstack((self.fitResults['fitResults']['x0'], self.fitResults['fitResults']['y0'], self.fitResults['tIndex'])).T

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

                else:
                    #default to time
                    self.glCanvas.setPoints(self.fitResults['fitResults']['x0'],self.fitResults['fitResults']['y0'],self.fitResults['tIndex'].astype('f'))
                    self.glCanvas.setCLim((0, self.numAnalysed))

        if (self.tq.getNumberOpenTasks(self.seriesName) + self.tq.getNumberTasksInProgress(self.seriesName)) == 0 and 'SpoolingFinished' in self.mdh.getEntryNames():
            self.dsviewer.statusbar.SetBackgroundColour(wx.GREEN)
            self.dsviewer.statusbar.Refresh()

        self.progPan.draw()
        self.progPan.Refresh()
        self.dsviewer.Refresh()
        self.dsviewer.update()

    def update(self):
        if 'fitInf' in dir(self) and not self.dsviewer.player.tPlay.IsRunning():
            self.fitInf.UpdateDisp(self.vp.view.PointsHitTest())


    #from fth5.py
    def checkTQ(self):
        if self.tq == None:
            if 'PYME_TASKQUEUENAME' in os.environ.keys():
                taskQueueName = os.environ['PYME_TASKQUEUENAME']
            else:
                taskQueueName = 'taskQueue'
            self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)


    def pushImages(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        if self.dataSource.moduleName == 'HDFDataSource':
            pushImagesHDF(startingAt, detThresh, fitFcn)
        else:
            pushImagesQueue(startingAt, detThresh, fitFcn)


    def pushImagesHDF(self, startingAt=0, detThresh = .9, fitFcn = 'LatGaussFitFR'):
        #global seriesName
        dataFilename = self.seriesName
        resultsFilename = genResultFileName(self.seriesName)
        while os.path.exists(resultsFilename):
            di, fn = os.path.split(resultsFilename)
            fdialog = wx.FileDialog(None, 'Analysis file already exists, please select a new filename',
                        wildcard='H5R files|*.h5r', defaultDir=di, defaultFile=os.path.splitext(fn)[0] + '_1.h5r', style=wx.SAVE)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                resultsFilename = fdialog.GetPath().encode()
            else:
                raise RuntimeError('Invalid results file - not running')
            self.seriesName = resultsFilename
        self.tq.createQueue('HDFTaskQueue', self.seriesName, dataFilename = dataFilename, resultsFilename=resultsFilename, startAt = 'notYet')
        mdhQ = MetaDataHandler.QueueMDHandler(self.tq, self.seriesName, self.mdh)
        mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
        mdhQ.setEntry('Analysis.FitModule', fitFcn)
        mdhQ.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.dataSource))
        evts = self.dataSource.getEvents()
        if len(evts) > 0:
            self.tq.addQueueEvents(self.seriesName, evts)
        self.tq.releaseTasks(self.seriesName, startingAt)


    def pushImagesQueue(self, startingAt=0, detThresh = .9, fitFcn='LatGaussFitFR'):
        self.mdh.setEntry('Analysis.DetectionThreshold', detThresh)
        self.mdh.setEntry('Analysis.FitModule', fitFcn)
        self.mdh.setEntry('Analysis.DataFileID', fileID.genDataSourceID(self.dataSource))
        self.tq.releaseTasks(self.seriesName, startingAt)


#    def testFrame(self, detThresh = 0.9):
#        ft = remFitBuf.fitTask(self.seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), cFitType.GetString(cFitType.GetSelection()), bgindices=range(max(vp.zp-10, self.mdh.getEntry('EstimatedLaserOnFrameNo')),vp.zp), SNThreshold=True)
#        return ft(True)
#
#    def testFrameTQ(self, detThresh = 0.9):
#        ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatGaussFitFR', 'TQDataSource', bgindices=range(max(vp.zp-10, mdh.getEntry('EstimatedLaserOnFrameNo')),vp.zp), SNThreshold=True)
#        return ft(True, tq)
#
#    def pushImagesD(self, startingAt=0, detThresh = .9):
#        self.tq.createQueue('HDFResultsTaskQueue', self.seriesName, None)
#        mdhQ = MetaDataHandler.QueueMDHandler(self.tq, self.seriesName, mdh)
#        mdhQ.setEntry('Analysis.DetectionThreshold', detThresh)
#        for i in range(startingAt, ds.shape[0]):
#            self.tq.postTask(remFitBuf.fitTask(self.seriesName,i, detThresh, MetaDataHandler.NestedClassMDHandler(self.mdh), 'LatGaussFitFR', bgindices=range(max(i-10,self.mdh.getEntry('EstimatedLaserOnFrameNo') ),i), SNThreshold=True,driftEstInd=range(max(i-5, self.mdh.getEntry('EstimatedLaserOnFrameNo')),min(i + 5, ds.shape[0])), dataSourceModule=self.dataSource.moduleName), queueName=self.seriesName)


#    def testFrameD(self, detThresh = 0.9):
#        ft = remFitBuf.fitTask(seriesName,vp.zp, detThresh, MetaDataHandler.NestedClassMDHandler(mdh), 'LatGaussFitFR', bgindices=range(max(vp.zp-10, md.EstimatedLaserOnFrameNo),vp.zp), SNThreshold=True,driftEstInd=range(max(vp.zp-5, md.EstimatedLaserOnFrameNo),min(vp.zp + 5, ds.shape[0])))
#        return ft(True)

    def testFrames(self, detThresh = 0.9, offset = 0):
        close('all')
        matplotlib.interactive(False)
        clf()
        sq = min(self.mdh.getEntry('EstimatedLaserOnFrameNo') + 1000, self.dataSource.getNumSlices()/4)
        zps = array(range(self.mdh.getEntry('EstimatedLaserOnFrameNo') + 20, self.mdh.getEntry('EstimatedLaserOnFrameNo') + 24)  + range(sq, sq + 4) + range(self.dataSource.getNumSlices()/2,self.dataSource.getNumSlices() /2+4))
        zps += offset
        fitMod = self.cFitType.GetStringSelection()
        #bgFrames = int(tBackgroundFrames.GetValue())
        bgFrames = [int(v) for v in self.tBackgroundFrames.GetValue().split(':')]
        for i in range(12):
            #if 'Analysis.NumBGFrames' in md.getEntryNames():
            #bgi = range(max(zps[i] - bgFrames,mdh.getEntry('EstimatedLaserOnFrameNo')), zps[i])
            bgi = range(max(zps[i] + bgFrames[0],self.mdh.getEntry('EstimatedLaserOnFrameNo')), max(zps[i] + bgFrames[1],self.mdh.getEntry('EstimatedLaserOnFrameNo')))
            #else:
            #    bgi = range(max(zps[i] - 10,md.EstimatedLaserOnFrameNo), zps[i])
            if 'Splitter' in fitMod:
                ft = remFitBuf.fitTask(self.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.mdh), 'SplitterObjFindR', bgindices=bgi, SNThreshold=True)
            else:
                ft = remFitBuf.fitTask(self.seriesName, zps[i], detThresh, MetaDataHandler.NestedClassMDHandler(self.mdh), 'LatObjFindFR', bgindices=bgi, SNThreshold=True)
            res = ft()
            xp = floor(i/4)/3.
            yp = (3 - i%4)/4.
            #print xp, yp
            axes((xp,yp, 1./6,1./4.5))
            #d = ds[zps[i], :,:].squeeze().T
            d = self.dataSource.getSlice(zps[i]).T
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
                    plot([p.x for p in ft.ofd], [d.shape[0] - p.y for p in ft.ofd], 'o', mew=2, mec='g', mfc='none', ms=9)
            #axis('tight')
            xlim(0, d.shape[1])
            ylim(0, d.shape[0])
            xticks([])
            yticks([])
        show()
        matplotlib.interactive(True)

