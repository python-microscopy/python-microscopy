#!/usr/bin/python

###############
# deconvDialogs.py
#
# Copyright David Baddeley, 2012
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
################
import wx
import os

LASTPSFFILENAME = ''

class DeconvSettingsDialog(wx.Dialog):
    def __init__(self, parent, beadMode=False, nChans=1):
        wx.Dialog.__init__(self, parent, title='Deconvolution')
        self.nChans = nChans

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        notebook = wx.Notebook(self, -1)

        #basic panel
        pan1 = wx.Panel(notebook, -1)
        notebook.AddPage(pan1, 'Basic')

        sizer2 = wx.BoxSizer(wx.VERTICAL)
        
        print(('nchans:', nChans))
        
        if nChans > 1:
            sizer3 = wx.BoxSizer(wx.HORIZONTAL)
            sizer3.Add(wx.StaticText(pan1, -1, 'Channel:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            self.cChannel = wx.Choice(pan1, -1, choices=['Chan %d' % i for i in range(nChans)])

            sizer3.Add(self.cChannel, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            sizer2.Add(sizer3, 0, wx.EXPAND | wx.ALL, 0)
        
        sizer4 = wx.StaticBoxSizer(wx.StaticBox(pan1, -1, 'PSF:'),wx.VERTICAL)
       

        if not beadMode:
            self.nb2 = wx.Notebook(pan1, -1)
            
            pan2 = wx.Panel(self.nb2, -1)
            self.nb2.AddPage(pan2, 'File')
            
            pan2.PSFMode = 'File'
            
            sizer3 = wx.BoxSizer(wx.HORIZONTAL)
            #sizer3.Add(wx.StaticText(pan2, -1, 'File:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            self.fpPSF = wx.FilePickerCtrl(pan2, -1, wildcard='*.psf|*.tif', style=wx.FLP_OPEN|wx.FLP_FILE_MUST_EXIST)
            self.fpPSF.Bind(wx.EVT_FILEPICKER_CHANGED, self.OnPSFFileChanged)

            sizer3.Add(self.fpPSF, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            pan2.SetSizerAndFit(sizer3)
            
            pan2 = wx.Panel(self.nb2, -1)
            self.nb2.AddPage(pan2, '2D Laplace')
            pan2.PSFMode = 'Laplace'
            s3 = wx.BoxSizer(wx.VERTICAL)
            s3.Add(wx.StaticText(pan2, -1, 'Used for 2D STED.'), 0, wx.ALL, 2)
            sizer3 = wx.BoxSizer(wx.HORIZONTAL)
            sizer3.Add(wx.StaticText(pan2, -1, 'FWHM [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            self.tLaplaceFWHM = wx.TextCtrl(pan2, -1, '50')

            sizer3.Add(self.tLaplaceFWHM, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            
            s3.Add(sizer3)
            pan2.SetSizerAndFit(s3)
            
#            pan2 = wx.Panel(nb2, -1)
#            nb2.AddPage(pan2, 'Widefield')
#            s3 = wx.BoxSizer(wx.VERTICAL)
#            s3.Add(wx.StaticText(pan2, -1, 'Using high NA scalar approx.'), 0, wx.ALL, 2)
#            
#            sizer3 = wx.BoxSizer(wx.HORIZONTAL)
#            sizer3.Add(wx.StaticText(pan2, -1, u'Emission \u03BB [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
#            self.tEmissionWavelength = wx.TextCtrl(pan2, -1, '700')
#            sizer3.Add(self.tEmissionWavelength, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)            
#            #s3.Add(sizer3, 0, wx.EXPAND, 0)
#            
#            #sizer3 = wx.BoxSizer(wx.HORIZONTAL)
#            sizer3.Add(wx.StaticText(pan2, -1, 'NA:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
#            self.tPSFNA = wx.TextCtrl(pan2, -1, '1.4')
#            sizer3.Add(self.tPSFNA, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)            
#            s3.Add(sizer3, 0, wx.EXPAND, 0)
#            
#            sizer3 = wx.BoxSizer(wx.HORIZONTAL)
#            sizer3.Add(wx.StaticText(pan2, -1, 'Abberations:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
#            self.tAbberations = wx.TextCtrl(pan2, -1, '{8:0,}')
#            sizer3.Add(self.tAbberations, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)            
#            s3.Add(sizer3, 0, wx.EXPAND, 0)
#            
#            s3.Add(wx.StaticText(pan2, -1, '(a list of Zernike modes and ammounts).\n 8 = SA, 4,5 = Astig, 6,7 = Coma'), 0, wx.ALL, 2)
#            
#            pan2.SetSizerAndFit(s3)
            
            sizer4.Add(self.nb2, 1, wx.EXPAND | wx.ALL, 0)
            self.nb2.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnPSFNotebookPageChanged)
            
            self.cbShowPSF = wx.CheckBox(pan1, -1, 'Show PSF')
            sizer4.Add(self.cbShowPSF, 0, wx.ALL, 2)
        else:
            sizer3 = wx.BoxSizer(wx.HORIZONTAL)
            sizer3.Add(wx.StaticText(pan1, -1, 'Bead Diameter [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            self.tBeadSize = wx.TextCtrl(pan1, -1, '200')

            sizer3.Add(self.tBeadSize, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            sizer4.Add(sizer3, 1, wx.EXPAND | wx.ALL, 0)

        sizer2.Add(sizer4, 0, wx.EXPAND | wx.ALL, 0)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Method:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.cMethod = wx.Choice(pan1, -1, choices=['ICTM', 'Richardson-Lucy'])
        self.cMethod.SetSelection(1)
        self.cMethod.Bind(wx.EVT_CHOICE, self.OnMethodChanged)

        sizer3.Add(self.cMethod, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        sizer2.Add(sizer3, 0, wx.EXPAND | wx.ALL, 0)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Number of iterations:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tNumIters = wx.TextCtrl(pan1, -1, '50')

        sizer3.Add(self.tNumIters, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND | wx.ALL, 0)
        
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Offset:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tOffset = wx.TextCtrl(pan1, -1, '0')

        sizer3.Add(self.tOffset, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND | wx.ALL, 0)
        
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Background:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tBackground = wx.TextCtrl(pan1, -1, '0')

        sizer3.Add(self.tBackground, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND | wx.ALL, 0)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, u'Regularisation \u03BB:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tRegLambda = wx.TextCtrl(pan1, -1, '1e-1')
        self.tRegLambda.Disable()

        sizer3.Add(self.tRegLambda, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND | wx.ALL, 0)

        pan1.SetSizerAndFit(sizer2)
        
        


        #blocking panel
        pan1 = wx.Panel(notebook, -1)
        notebook.AddPage(pan1, 'Blocking')

        sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.cbBlocking = wx.CheckBox(pan1, -1, 'Do tiled/blocked deconvolution')
        sizer2.Add(self.cbBlocking, 0,  wx.ALL, 5)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Tile size:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tTileSize = wx.TextCtrl(pan1, -1, '128')

        sizer3.Add(self.tTileSize, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.ALL, 0)

        pan1.SetSizerAndFit(sizer2)

        #padding panel
        pan1 = wx.Panel(notebook, -1)
        notebook.AddPage(pan1, 'Padding')

        sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.cbPadding = wx.CheckBox(pan1, -1, 'Pad data')
        sizer2.Add(self.cbPadding, 0, wx.ALL, 5)

        self.cbRemovePadding = wx.CheckBox(pan1, -1, 'Remove padding on completion')
        sizer2.Add(self.cbRemovePadding, 0, wx.ALL, 5)
        self.cbRemovePadding.SetValue(True)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Pad width:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tPadWidth = wx.TextCtrl(pan1, -1, '30,30,10')

        sizer3.Add(self.tPadWidth, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.ALL, 0)

        pan1.SetSizerAndFit(sizer2)

        sizer1.Add(notebook, 1, wx.EXPAND|wx.ALL, 5)

        btSizer = wx.StdDialogButtonSizer()

        self.bOK = wx.Button(self, wx.ID_OK)
        if not beadMode:
            if os.path.exists(LASTPSFFILENAME):
                 self.fpPSF.SetPath(LASTPSFFILENAME)
            else:
                self.bOK.Disable()
        self.bOK.SetDefault()

        btSizer.AddButton(self.bOK)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def GetNumIterationss(self):
        return int(self.tNumIters.GetValue())

    def GetRegularisationLambda(self):
        return float(self.tRegLambda.GetValue())

    def GetMethod(self):
        return self.cMethod.GetStringSelection()

    def GetBlocking(self):
        return self.cbBlocking.GetValue()

    def GetBlockSize(self):
        return int(self.tTileSize.GetValue())

    def GetPadding(self):
        return self.cbPadding.GetValue()

    def GetRemovePadding(self):
        return self.cbRemovePadding.GetValue()

    def GetPadSize(self):
        return [int(w) for w in self.tPadWidth.GetValue().split(',')]

    def GetBeadRadius(self):
        return float(self.tBeadSize.GetValue())/2

    def GetPSFFilename(self):
        return self.fpPSF.GetPath()

    def OnPSFFileChanged(self, event):
        global LASTPSFFILENAME
        
        if os.path.exists(self.fpPSF.GetPath()):
            LASTPSFFILENAME = self.fpPSF.GetPath()
            self.bOK.Enable()
            
    def OnPSFNotebookPageChanged(self, event):    
        if os.path.exists(LASTPSFFILENAME) or not event.GetSelection() == 0:#self.nb2.GetCurrentPage().PSFMode == 'File':
            self.bOK.Enable()
        else:
            self.bOK.Disable()
            
        event.Skip()
        
    def OnMethodChanged(self, event):    
        if self.cMethod.GetStringSelection() == 'ICTM':
            self.tRegLambda.Enable()
        else:
            self.tRegLambda.Disable()
            
        event.Skip()
        
    def GetOffset(self):
        return float(self.tOffset.GetValue())
        
    def GetBackground(self):
        return float(self.tBackground.GetValue())
        
    def GetChannel(self):
        if self.nChans == 1:
            return 0
        else:
            return self.cChannel.GetSelection()
            
    def GetPSF(self, vshint = None):
        import numpy as np
        from PYME.IO.load_psf import load_psf
        from scipy import stats
        
        PSFMode = self.nb2.GetCurrentPage().PSFMode
        #get PSF from file
        if PSFMode == 'File':
            fn = self.GetPSFFilename()
            psf, vs = load_psf(fn)
            psf = np.atleast_3d(psf)
            
            return (fn, psf, vs)
        elif (PSFMode == 'Laplace'):
            sc = float(self.tLaplaceFWHM.GetValue())/2.0
            X, Y = np.mgrid[-30.:31., -30.:31.]
            R = np.sqrt(X*X + Y*Y)
            
            if not vshint is None:
                vx = vshint
            else:
                vx = sc/2.
            
            vs = type('vs', (object,), dict(x=vx, y=vx))
            
            psf = np.atleast_3d(stats.cauchy.pdf(vx*R, scale=sc))
                
            return 'Generated Laplacian, FWHM=%f' % (2*sc), psf/psf.sum(), vs
            


class DeconvProgressDialog(wx.Dialog):
    def __init__(self, parent, numIters):
        wx.Dialog.__init__(self, parent, title='Deconvolution Progress')
        self.cancelled = False

        self.numIters = numIters

        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.gProgress = wx.Gauge(self, -1, numIters)

        sizer1.Add(self.gProgress, 0, wx.EXPAND | wx.ALL, 5)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_CANCEL)
        btn.Bind(wx.EVT_BUTTON, self.OnCancel)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnCancel(self, event):
        self.cancelled = True
        #self.EndModal(wx.ID_CANCEL)

    def Tick(self, dec):
        if not self.cancelled:
            self.gProgress.SetValue(dec.loopcount)
            return True
        else:
            return False

class DeconvProgressPanel(wx.Panel):
    def __init__(self, parent, numIters):
        wx.Panel.__init__(self, parent)
        self.cancelled = False

        self.numIters = numIters

        sizer1 = wx.BoxSizer(wx.HORIZONTAL)

        self.gProgress = wx.Gauge(self, -1, numIters)

        sizer1.Add(self.gProgress, 5, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        #btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_CANCEL)
        btn.Bind(wx.EVT_BUTTON, self.OnCancel)

        #btSizer.AddButton(btn)

        #btSizer.Realize()

        sizer1.Add(btn, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnCancel(self, event):
        self.cancelled = True
        #self.EndModal(wx.ID_CANCEL)

    def Tick(self, dec):
        if not self.cancelled:
            self.gProgress.SetValue(dec.loopcount)
            return True
        else:
            return False

    

    
   