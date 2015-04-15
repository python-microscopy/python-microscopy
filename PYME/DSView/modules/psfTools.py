#!/usr/bin/python
##################
# coloc.py
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
#import numpy
import wx
import wx.grid
#import pylab
#from PYME.DSView.image import ImageStack
try:
    from enthought.traits.api import HasTraits, Float, Int
    from enthought.traits.ui.api import View, Item
    from enthought.traits.ui.menu import OKButton
except ImportError:
    from traits.api import HasTraits, Float, Int
    from traitsui.api import View, Item
    from traitsui.menu import OKButton

from graphViewPanel import *
from PYME.PSFEst import psfQuality

def remove_newlines(s):
    s = '<>'.join(s.split('\n\n'))
    s = ' '.join(s.split())
    return '\n'.join(s.split('<>'))

class PSFQualityPanel(wx.Panel):
    def __init__(self, dsviewer):
        wx.Panel.__init__(self, dsviewer) 
        
        self.image = dsviewer.image
        self.dsviewer = dsviewer
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.grid = wx.grid.Grid(self, -1)
        self.grid.CreateGrid(len(psfQuality.test_names), 2)
        self.grid.EnableEditing(0)
        
        self.grid.SetColLabelValue(0, "Localisation")
        self.grid.SetColLabelValue(1, "Deconvolution")
        
        for i, testName in enumerate(psfQuality.test_names):
            self.grid.SetRowLabelValue(i, testName)
        
        self.FillGrid()
        
        self.grid.AutoSizeColumns()
        self.grid.SetRowLabelSize(wx.grid.GRID_AUTOSIZE)
        
        vsizer.Add(self.grid, 2, wx.EXPAND|wx.ALL, 5)
        vsizer.Add(wx.StaticText(self, -1, 'Click a cell for description'), 0, wx.ALL, 5)
        
        self.description = wx.TextCtrl(self, -1, '', style = wx.TE_MULTILINE|wx.TE_AUTO_SCROLL|wx.TE_READONLY)
        vsizer.Add(self.description, 1, wx.EXPAND|wx.ALL, 5)
        
        self.grid.Bind(wx.grid.EVT_GRID_CELL_LEFT_CLICK, self.OnSelectCell)
        
        self.SetSizerAndFit(vsizer)
        
    def OnSelectCell(self, event):
        r = event.GetRow()
        c = event.GetCol()
        
        self.description.SetValue('')
        
        name = psfQuality.test_names[r]
        
        if c == 0:
            #localisaitons
            try:
                self.description.SetValue(remove_newlines(psfQuality.localisation_tests[name].__doc__))
            except KeyError:
                pass
        elif c == 1:
            #deconvolution
            try:
                self.description.SetValue(remove_newlines(psfQuality.deconvolution_tests[name].__doc__))
            except KeyError:
                pass
            
        event.Skip()
        
    def FillGrid(self, caller=None):
        loc_res, dec_res = psfQuality.runTests(self.image, self.dsviewer.crbv)
        
        for i, testName in enumerate(psfQuality.test_names):
            try:
                val, merit = loc_res[testName]
                colour = psfQuality.colour(merit)
                self.grid.SetCellValue(i, 0, '%3.3g' % val)
                self.grid.SetCellBackgroundColour(i, 0, tuple(colour*255))
            except KeyError:
                pass
            
            try:
                val, merit = dec_res[testName]
                colour = psfQuality.colour(merit)
                self.grid.SetCellValue(i, 1, '%3.3g' % val)
                self.grid.SetCellBackgroundColour(i, 1, tuple(colour*255))
            except KeyError:
                pass
        

class CRBViewPanel(wx.Panel):
    def __init__(self, parent, image, background=1):
        wx.Panel.__init__(self, parent)
        
        self.image = image
        self.background = background
             
        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)

        sizer1.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Background photons:', pos = (0,0)), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBackground = wx.TextCtrl(self, -1, '%d' % self.background, pos=(0, 0))
        hsizer.Add(self.tBackground, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBackground.Bind(wx.EVT_TEXT, self.OnChangeBackground)
        
        sizer1.Add(hsizer)

        self.Bind(wx.EVT_SIZE, self._onSize)

        #self.toolbar.update()
        self.SetSizerAndFit(sizer1)
        self.calcCRB()
        #self.draw()
        
    def OnChangeBackground(self, event):
        print('b')
        self.background = float(self.tBackground.GetValue())
        self.calcCRB()
        
    def calcCRB(self, caller=None):
        from PYME.Analysis import cramerRao
        from PYME.PSFGen import fourierHNA
        #print 'b'
        import numpy as np
        d = self.image.data[:,:,:]
        I = d[:,:,d.shape[2]/2].sum()
        
        vs = 1e3*np.array([self.image.mdh['voxelsize.x'], self.image.mdh['voxelsize.y'],self.image.mdh['voxelsize.z']])
        
        #print 'fi'        
        FI = cramerRao.CalcFisherInformZn2(d*(2e3/I) + self.background, 100, voxelsize=vs)
        #print 'crb'
        self.crb = cramerRao.CalcCramerReoZ(FI)
        #print 'crbd'
        
        z_ = np.arange(d.shape[2])*self.image.mdh['voxelsize.z']*1.0e3
        self.z_ = z_ - z_.mean()
        
        ps_as = fourierHNA.GenAstigPSF(self.z_, vs[0], 2)  
        I = ps_as[:,:,ps_as.shape[2]/2].sum()
        self.crb_as = (cramerRao.CalcCramerReoZ(cramerRao.CalcFisherInformZn2(ps_as*2000/I + self.background, 500, voxelsize=vs)))
        
        self.draw()


    def draw(self, event=None):
        self.axes.cla()

        self.axes.plot(self.z_, np.sqrt(self.crb[:,0]), label='x')
        self.axes.plot(self.z_, np.sqrt(self.crb[:,1]), label='y')
        self.axes.plot(self.z_, np.sqrt(self.crb[:,2]), label='z')
        self.axes.legend()
        
        self.axes.set_xlabel('Defocus [nm]')
        self.axes.set_ylabel('Std. Dev. [nm]')
        self.axes.set_title('Cramer-Rao bound for 2000 photons')

        crb_as = np.sqrt(self.crb_as)        
        
        self.axes.plot(self.z_, crb_as[:,0], 'b:')
        self.axes.plot(self.z_, crb_as[:,1], 'g:')
        self.axes.plot(self.z_, crb_as[:,2], 'r:')

        self.canvas.draw()

    def _onSize( self, event ):
        #self._resizeflag = True
        self._SetSize()


    def _SetSize( self ):
        pixels = tuple( self.GetClientSize() )
        self.SetSize( pixels )
        self.canvas.SetSize( pixels )
        self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                     float( pixels[1] )/self.figure.get_dpi() )


class PSFTools(HasTraits):
    wavelength = Float(700)
    NA = Float(1.49)
    pupilSize = Float(0)
    iterations = Int(50)
    
    view = View(Item('wavelength'),
                Item('NA'),
                Item('pupilSize'),
                Item('iterations'), buttons=[OKButton])
    
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        dsviewer.AddMenuItem('Processing', "Extract &Pupil Function", self.OnExtractPupil)
        dsviewer.AddMenuItem('Processing', "Cramer-Rao Bound vs Background ", self.OnCalcCRB3DvsBG)
        dsviewer.AddMenuItem('Processing', "PSF Background Correction", self.OnSubtractBackground)
        #wx.EVT_MENU(dsviewer, PROC_LABEL, self.OnLabel)

    def OnExtractPupil(self, event):
        import numpy as np
        import pylab
        from PYME.PSFGen import fourierHNA
        
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D

        z_ = np.arange(self.image.data.shape[2])*self.image.mdh['voxelsize.z']*1.e3
        z_ -= z_.mean()  
        
        self.configure_traits(kind='modal')
        
        pupil = fourierHNA.ExtractPupil(np.maximum(self.image.data[:,:,:] - .001, 0), z_, self.image.mdh['voxelsize.x']*1e3, self.wavelength, self.NA, nIters=self.iterations, size=self.pupilSize)
        
        pylab.figure()
        pylab.subplot(121)
        pylab.imshow(np.abs(pupil), interpolation='nearest')
        pylab.subplot(122)
        pylab.imshow(np.angle(pupil)*(np.abs(pupil) > 0), interpolation='nearest')
        
        pupil = pupil*(np.abs(pupil) > 0)
        
        #im = ImageStack([np.abs(pupil), np.angle(pupil)*(np.abs(pupil) > 0)], titleStub = 'Extracted Pupil')
        im = ImageStack(pupil, titleStub = 'Extracted Pupil')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi
        mode = 'pupil'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
        
        
    def OnSubtractBackground(self, event):
        from PYME.DSView.image import ImageStack
        from PYME.DSView import ViewIm3D
        from PYME.PSFEst import extractImages

        d_bg = extractImages.backgroundCorrectPSFWF(self.image.data[:,:,:])
        
        
        
        im = ImageStack(d_bg, titleStub = 'Filtered Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        
        dv = ViewIm3D(im, mode='psf', glCanvas=self.dsviewer.glCanvas)

        
    def OnCalcCRB(self, event):
        #print 'f'
        from PYME.Analysis import cramerRao
        from PYME.PSFGen import fourierHNA
        #print 'b'
        import numpy as np
        
        d = self.image.data[:,:,:]
        I = d[:,:,d.shape[2]/2].sum()
        
        vs = 1e3*np.array([self.image.mdh['voxelsize.x'], self.image.mdh['voxelsize.y'],self.image.mdh['voxelsize.z']])
        
        #print 'fi'        
        FI = cramerRao.CalcFisherInformZn2(d*(2e3/I), 100, voxelsize=vs)
        #print 'crb'
        crb = cramerRao.CalcCramerReoZ(FI)
        #print 'crbd'
        
        import pylab
        z_ = np.arange(d.shape[2])*self.image.mdh['voxelsize.z']*1.0e3
        z_ = z_ - z_.mean()
        
        print('p')
        pylab.figure()
        pylab.plot(z_, np.sqrt(crb[:,0]), label='x')
        pylab.plot(z_, np.sqrt(crb[:,1]), label='y')
        pylab.plot(z_, np.sqrt(crb[:,2]), label='z')
        pylab.legend()
        
        pylab.xlabel('Defocus [nm]')
        pylab.ylabel('Std. Dev. [nm]')
        pylab.title('Cramer-Rao bound for 2000 photons')
        
        ps_as = fourierHNA.GenAstigPSF(z_, vs[0], 2)  
        I = ps_as[:,:,ps_as.shape[2]/2].sum()
        crb_as = np.sqrt(cramerRao.CalcCramerReoZ(cramerRao.CalcFisherInformZn2(ps_as*2000/I, 500, voxelsize=vs)))
        pylab.plot(z_, crb_as[:,0], 'b:')
        pylab.plot(z_, crb_as[:,1], 'g:')
        pylab.plot(z_, crb_as[:,2], 'r:')
        
        
    def OnCalcCRB3DvsBG(self, event):
        from PYME.Analysis import cramerRao
        from PYME.PSFGen import fourierHNA
        #print 'b'
        import numpy as np
        
        
        vs = 1e3*np.array([self.image.mdh['voxelsize.x'], self.image.mdh['voxelsize.y'],self.image.mdh['voxelsize.z']])
        
        zf = self.image.data.shape[2]/2
        dz = 500/vs[2]

        d = self.image.data[:,:,(zf-dz):(zf + dz + 1)]
        I = d[:,:,d.shape[2]/2].sum()        
        
        
        bgv = np.logspace(-1, 2)
        z_ = np.arange(d.shape[2])*vs[2]
        z_ = z_ - z_.mean()
        
        ps_as = fourierHNA.GenAstigPSF(z_, vs[0], 2)
        Ias = ps_as[:,:,ps_as.shape[2]/2].sum()
        
        crb3D = []
        crb3Das = []
        
        for bg in bgv:
            FI = cramerRao.CalcFisherInformZn2(d*(2e3/I) + bg, 100, voxelsize=vs)
            crb = cramerRao.CalcCramerReoZ(FI)
            crb_as = (cramerRao.CalcCramerReoZ(cramerRao.CalcFisherInformZn2(ps_as*2000/Ias + bg, 500, voxelsize=vs)))
            
            crb3D.append(np.sqrt(crb.sum(1)).mean())
            crb3Das.append(np.sqrt(crb_as.sum(1)).mean())
            
        import pylab
        
        pylab.figure()
        pylab.plot(bgv, crb3Das, label='Theoretical PSF')
        pylab.plot(bgv, crb3D, label='Measured PSF')
        pylab.legend()
        
        pylab.xlabel('Background [photons]')
        pylab.ylabel('Average CRB 3D')
        pylab.title('Cramer-Rao bound vs Background')
        

    



def Plug(dsviewer):
    dsviewer.PSFTools = PSFTools(dsviewer)
    if dsviewer.do.ds.shape[2] > 1:
        dsviewer.crbv = CRBViewPanel(dsviewer, dsviewer.image)
        dsviewer.dataChangeHooks.append(dsviewer.crbv.calcCRB)
        
        dsviewer.psfqp = PSFQualityPanel(dsviewer)
        dsviewer.dataChangeHooks.append(dsviewer.psfqp.FillGrid)
        
        #dsviewer.AddPage(dsviewer.psfqp, False, 'PSF Quality')
        dsviewer.AddPage(dsviewer.crbv, False, 'Cramer-Rao Bounds')
            
        pinfo1 = aui.AuiPaneInfo().Name("psfQPanel").Left().Caption('PSF Quality').DestroyOnClose(True).CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(dsviewer.psfqp, pinfo1)
        
    dsviewer._mgr.Update()



