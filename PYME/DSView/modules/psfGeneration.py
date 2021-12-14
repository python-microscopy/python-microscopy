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

from PYME.recipes.traits import HasTraits, Float, Int, List, Instance, Property

from PYME.Analysis.PSFGen import fourierHNA
from PYME.IO import image
from PYME.simulation.pointsets import WRDictEnum
from PYME.misc import zernike

class Pupil(HasTraits):
    pass

class WidefieldPupil(Pupil):
    def GeneratePupil(self, pixelSize, size, wavelength, NA, n):
        return fourierHNA.widefield_pupil_and_propagator(pixelSize, size, wavelength, NA, n)
    
class MeasuredPupil(Pupil):
    image = WRDictEnum(image.openImages)
    
    def GeneratePupil(self, pixelSize, size, wavelength, NA, n):
        return image.openImages[self.image].data[:,:]
        
class PhaseRampPupil(Pupil):
    strength = Float(0.5)
    def GeneratePupil(self, pixelSize, size, wavelength, NA, n):
        return fourierHNA.GenPRPupil(pixelSize, size, wavelength, NA, n, self.strength)
        
class ZernikeMode(HasTraits):
    num = Int(0)
    coeff = Float(0)
    name = Property()
    
    def _get_name(self):
        return zernike.NameByNumber[self.num]
    
    def __init__(self, num):
        self.num = num

_pupils = [WidefieldPupil(), MeasuredPupil()]
def _getDefaultPupil():
    return _pupils[0]

class PupilGenerator(HasTraits):
    wavelength = Float(700)
    NA = Float(1.49)
    n = Float(1.51)
    pupilSizeX = Int(61)
    pixelSize = Float(70)
    
    pupils = List(_pupils)
    
    aberations = List(ZernikeMode, value=[ZernikeMode(i) for i in range(25)])

    basePupil = Instance(Pupil, _getDefaultPupil)
    
    pupil = None
    
    def default_traits_view( self ):
        from traitsui.api import View, Item, InstanceEditor
        from traitsui.menu import OKButton
        
        view = View(Item( 'basePupil',
                            label= 'Pupil source',
                            editor =
                            InstanceEditor(name = 'pupils',
                                editable = True),
                                ),
                Item('_'),
                Item('wavelength'),
                Item('n'),
                Item('NA'),
                Item('pupilSizeX'),
                Item('pixelSize'),
                Item('_'),
                Item('aberations'),
                buttons=[OKButton])
        
        return view
                
    def GetPupil(self):
        u, v, R, pupil = self.basePupil.GeneratePupil(self.pixelSize, self.pupilSizeX, self.wavelength, self.NA, self.n)
        
    
    
        

    



def Plug(dsviewer):
    dsviewer.PSFGen = PSFG(dsviewer)
    
    dsviewer.crbv = CRBViewPanel(dsviewer, dsviewer.image)
    dsviewer.dataChangeHooks.append(dsviewer.crbv.calcCRB)
    
    dsviewer.psfqp = PSFQualityPanel(dsviewer)
    dsviewer.dataChangeHooks.append(dsviewer.psfqp.FillGrid)
    
    #dsviewer.AddPage(dsviewer.psfqp, False, 'PSF Quality')
    dsviewer.AddPage(dsviewer.crbv, False, 'Cramer-Rao Bounds')
    
    
    #dsviewer.gv.toolbar = MyNavigationToolbar(dsviewer.gv.canvas, dsviewer)
    #dsviewer._mgr.AddPane(dsviewer.gv.toolbar, aui.AuiPaneInfo().Name("MPLTools").Caption("Matplotlib Tools").CloseButton(False).
    #                  ToolbarPane().Right().GripperTop())
    
    pinfo1 = aui.AuiPaneInfo().Name("psfQPanel").Left().Caption('PSF Quality').DestroyOnClose(True).CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
    dsviewer._mgr.AddPane(dsviewer.psfqp, pinfo1)
    dsviewer._mgr.Update()



