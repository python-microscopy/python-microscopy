#!/usr/bin/python
##################
# blobFinding.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import PYME.misc.autoFoldPanel as afp


class blobFinder:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.dataSource = dsviewer.dataSource
        self.mdh = dsviewer.mdh

        self.vObjPos = None
        self.vObjFit = None

        F_SAVE_POSITIONS = wx.NewId()
        F_SAVE_FITS = wx.NewId()

        dsviewer.save_menu.Append(F_SAVE_POSITIONS, "Save &Positions", "", wx.ITEM_NORMAL)
        dsviewer.save_menu.Append(F_SAVE_FITS, "Save &Fit Results", "", wx.ITEM_NORMAL)

        wx.EVT_MENU(dsviewer, F_SAVE_POSITIONS, self.savePositions)
        wx.EVT_MENU(dsviewer, F_SAVE_FITS, self.saveFits)

    def GenBlobFindingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Object Finding", pinned = True)

        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tThreshold = wx.TextCtrl(pan, -1, value='50', size=(40, -1))

        hsizer.Add(self.tThreshold, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)

        _pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        self.cbSNThreshold = wx.CheckBox(item, -1, 'SNR Threshold')
        self.cbSNThreshold.SetValue(False)

        _pnl.AddFoldPanelWindow(item, self.cbSNThreshold, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)

        bFindObjects = wx.Button(item, -1, 'Find')


        bFindObjects.Bind(wx.EVT_BUTTON, self.OnFindObjects)
        #_pnl.AddFoldPanelWindow(item, bFindObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(pan)
        _pnl.AddPane(item)

    def OnFindObjects(self, event):
        threshold = float(self.tThreshold.GetValue())

        from PYME.Analysis.ofind3d import ObjectIdentifier

        if not 'ofd' in dir(self):
            #create an object identifier
            self.ofd = ObjectIdentifier(self.dataSource)

        #and identify objects ...
        if self.cbSNThreshold.GetValue(): #don't detect objects in poisson noise
            fudgeFactor = 1 #to account for the fact that the blurring etc... in ofind doesn't preserve intensities - at the moment completely arbitrary so a threshold setting of 1 results in reasonable detection.
            threshold =  (numpy.sqrt(self.mdh.Camera.ReadNoise**2 + numpy.maximum(self.mdh.Camera.ElectronsPerCount*(self.mdh.Camera.NoiseFactor**2)*(self.dataSource.astype('f') - self.mdh.Camera.ADOffset)*self.mdh.Camera.TrueEMGain, 1))/self.mdh.Camera.ElectronsPerCount)*fudgeFactor*threshold
            self.ofd.FindObjects(threshold, 0)
        else:
            self.ofd.FindObjects(threshold)

        self.vp.points = numpy.array([[p.x, p.y, p.z] for p in self.ofd])

        self.objPosRA = numpy.rec.fromrecords(self.vp.points, names='x,y,z')

        if self.vObjPos == None:
            self.vObjPos = recArrayView.recArrayPanel(self.notebook1, self.objPosRA)
            self.notebook1.AddPage(self.vObjPos, 'Object Positions')
        else:
            self.vObjPos.grid.SetData(self.objPosRA)

        self.dsviewer.update()

    def GenBlobFitPanel(self):
        item = afp.foldingPane(_pnl, -1, caption="Object Fitting", pinned = True)
#        item = _pnl.AddFoldPanel("Object Fitting", collapsed=False,
#                                      foldIcons=self.Images)

        bFitObjects = wx.Button(item, -1, 'Fit')


        bFitObjects.Bind(wx.EVT_BUTTON, self.OnFitObjects)
        #_pnl.AddFoldPanelWindow(item, bFitObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(bFitObjects)
        _pnl.AddPane(item)

    def OnFitObjects(self, event):
        import PYME.Analysis.FitFactories.Gauss3DFitR as fitMod

        fitFac = fitMod.FitFactory(self.dataSource, self.mdh)

        self.objFitRes = numpy.empty(len(self.ofd), fitMod.FitResultsDType)
        for i in range(len(self.ofd)):
            p = self.ofd[i]
            try:
                self.objFitRes[i] = fitFac.FromPoint(round(p.x), round(p.y), round(p.z))
            except:
                pass


        if self.vObjFit == None:
            self.vObjFit = recArrayView.recArrayPanel(self.dsviewer.notebook1, self.objFitRes['fitResults'])
            self.dsviewer.notebook1.AddPage(self.vObjFit, 'Fitted Positions')
        else:
            self.vObjFit.grid.SetData(self.objFitRes)

        self.dsviewer.update()

    def savePositions(self, event=None):
        fdialog = wx.FileDialog(None, 'Save Positions ...',
            wildcard='Tab formatted text|*.txt', defaultFile=os.path.splitext(self.seriesName)[0] + '_pos.txt', style=wx.SAVE|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath().encode()

            of = open(outFilename, 'w')
            of.write('\t'.join(self.objPosRA.dtype.names) + '\n')

            for obj in self.objPosRA:
                of.write('\t'.join([repr(v) for v in obj]) + '\n')
            of.close()

            npFN = os.path.splitext(outFilename)[0] + '.npy'

            numpy.save(npFN, self.objPosRA)

    def saveFits(self, event=None):
        fdialog = wx.FileDialog(None, 'Save Fit Results ...',
            wildcard='Tab formatted text|*.txt', defaultFile=os.path.splitext(self.seriesName)[0] + '_fits.txt', style=wx.SAVE|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath().encode()

            of = open(outFilename, 'w')
            of.write('\t'.join(self.objFitRes['fitResults'].dtype.names) + '\n')

            for obj in self.objFitRes['fitResults']:
                of.write('\t'.join([repr(v) for v in obj]) + '\n')
            of.close()

            npFN = os.path.splitext(outFilename)[0] + '.npy'

            numpy.save(npFN, self.objFitRes)
