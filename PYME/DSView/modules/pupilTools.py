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
#from PYME.IO.image import ImageStack
from six.moves import xrange

from PYME.recipes.traits import HasTraits, Float, Int, CStr, Bool

class ZernikeView(wx.ScrolledWindow):
    def __init__(self, dsviewer):
        from PYME.misc import zernike
        import numpy as np
        
        self.dsviewer = dsviewer
        mag = np.abs(dsviewer.image.data[:,:])
        phase = np.angle(dsviewer.image.data[:,:])
        
        xm = np.where(mag.max(1) > 0)[0]
        ym = np.where(mag.max(0) > 0)[0]
        
        print((xm, ym, mag.shape))

        mag = mag[xm[0]:(xm[-1]+1), ym[0]:(ym[-1]+1)]        
        phase = phase[xm[0]:(xm[-1]+1), ym[0]:(ym[-1]+1)]
        
        #im = mag*np.exp(1j*phase)
        
        coeffs, res, im = zernike.calcCoeffs(phase, 25, mag)
        
        #s = ''
        dsviewer.zernModes = {i : c for i, c in enumerate(coeffs)}
        #for i, c, r, in zip(xrange(25), coeffs, res):
        #    s += '%d\t%s%3.3f\tresidual=%3.2f\n' % (i, zernike.NameByNumber[i].ljust(30), c, r)
        
        wx.ScrolledWindow.__init__(self, dsviewer)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.gZern = wx.grid.Grid(self, -1)
        self.gZern.CreateGrid(len(coeffs), 3)
        self.gZern.EnableEditing(0)
        
        self.gZern.SetColLabelValue(0, "Name")
        self.gZern.SetColLabelValue(1, "Coefficient")
        self.gZern.SetColLabelValue(2, "Residual")
        
        for i, c, r, in zip(xrange(len(coeffs)), coeffs, res):
            self.gZern.SetRowLabelValue(i, '%d' %i)
            self.gZern.SetCellValue(i, 0, zernike.NameByNumber[i])
            self.gZern.SetCellValue(i, 1, '%3.3g' % c)
            self.gZern.SetCellValue(i, 2, '%3.3g' % r)
            
            cv = 255 - 125*(abs(c))
            col = (255, cv, cv)
            
            self.gZern.SetCellBackgroundColour(i, 0, col)
            self.gZern.SetCellBackgroundColour(i, 1, col)
            self.gZern.SetCellBackgroundColour(i, 2, col)
            
        self.gZern.AutoSizeColumns()
        self.gZern.SetRowLabelSize(wx.grid.GRID_AUTOSIZE)
        
        vsizer.Add(self.gZern, 1, wx.EXPAND|wx.ALL, 5)
        
        #vsizer.Add(wx.StaticText(self, -1, s), 1, wx.EXPAND)
        
        self.SetSizer(vsizer)
        self.SetScrollRate(0, 10)
        
        

class PupilTools(HasTraits):
    wavelength = Float(700)
    NA = Float(1.49)
    sizeX = Int(61)
    sizeZ = Int(61)
    zSpacing = Float(50) 
    apodization = CStr('sine')
    vectorial = Bool(False)
    

    def default_traits_view( self ):
        from traitsui.api import View, Item
        from traitsui.menu import OKButton
        
        view = View(Item('wavelength'),
                    Item('NA'),
                    Item('zSpacing'),
                    Item('sizeZ'),
                    Item('sizeX'),
                    Item('apodization'),
                    Item('vectorial'),
                    buttons=[OKButton])
        
        return view

    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        dsviewer.AddMenuItem('Processing', "Generate PSF from pupil", self.OnPSFFromPupil)
        dsviewer.AddMenuItem('Processing', "Generate PSF from Zernike modes", self.OnPSFFromZernikeModes)

    def OnPSFFromPupil(self, event):
        import numpy as np
        #import pylab
        from PYME.Analysis.PSFGen import fourierHNA
        
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        
        self.configure_traits(kind='modal')

        z_ = np.arange(self.sizeZ)*float(self.zSpacing)
        z_ -= z_.mean()        
        
        if self.vectorial:
            ps = fourierHNA.PsfFromPupilVect(self.image.data[:,:], z_, self.image.voxelsize_nm.x, lamb=self.wavelength, apodization=self.apodization, NA=self.NA)#, shape = [self.sizeX, self.sizeX])
            #ps = abs(ps*np.conj(ps))
        else:
            ps = fourierHNA.PsfFromPupil(self.image.data[:,:], z_, self.image.voxelsize_nm.x, lamb=self.wavelength, apodization=self.apodization, NA=self.NA)#, shape = [self.sizeX, self.sizeX])
        
        #ps = ps/ps[:,:,self.sizeZ/2].sum()
        
        ps = ps/ps.max()
        
        im = ImageStack(ps, titleStub = 'Generated PSF')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi
        mode = 'psf'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

    def OnPSFFromZernikeModes(self, event):
        import numpy as np
        #import pylab
        from PYME.Analysis.PSFGen import fourierHNA
        
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        
        self.configure_traits(kind='modal')

        z_ = np.arange(self.sizeZ)*float(self.zSpacing)
        z_ -= z_.mean()        
        
        #if self.vectorial:
        #    ps = fourierHNA.PsfFromPupilVect(self.image.data[:,:], z_, self.image.mdh['voxelsize.x']*1e3, self.wavelength, apodization=self.apodization, NA=self.NA)#, shape = [self.sizeX, self.sizeX])
        #    #ps = abs(ps*np.conj(ps))
        #else:
        #    ps = fourierHNA.PsfFromPupil(self.image.data[:,:], z_, self.image.mdh['voxelsize.x']*1e3, self.wavelength, apodization=self.apodization, NA=self.NA)#, shape = [self.sizeX, self.sizeX])
        
        ps = fourierHNA.GenZernikeDPSF(z_, dx = self.image.voxelsize_nm.x,
                                       zernikeCoeffs = self.dsviewer.zernModes, lamb=self.wavelength, 
                                       n=1.51, NA = self.NA, ns=1.51, beadsize=0, 
                                       vect=self.vectorial, apodization=self.apodization)
        #ps = ps/ps[:,:,self.sizeZ/2].sum()
        
        ps = ps/ps.max()
        
        im = ImageStack(ps, titleStub = 'Generated PSF')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.CropROI'] = roi
        mode = 'psf'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))     

    



def Plug(dsviewer):
    dsviewer.PupilTools = PupilTools(dsviewer)
    dsviewer.zern = ZernikeView(dsviewer)
    dsviewer.AddPage(dsviewer.zern, False, 'Zernike Moments')



