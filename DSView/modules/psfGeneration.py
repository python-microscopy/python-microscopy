#!/usr/bin/python
##################
# coloc.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
#import numpy
import wx
import wx.grid
#import pylab
#from PYME.DSView.image import ImageStack
from enthought.traits.api import HasTraits, Float, Int, List, Instance, Property
from enthought.traits.ui.api import View, Item, InstanceEditor
from enthought.traits.ui.menu import OKButton

from PYME.PSFGen import fourierHNA
from PYME.DSView import image
from PYME.Analysis.LMVis.Extras.pointSetGeneration import WRDictEnum
from PYME.misc import zernike

class Pupil(HasTraits):
    pass

class WidefieldPupil(Pupil):
    def GeneratePupil(self, pixelSize, size, wavelength, NA, n):
        return fourierHNA.GenWidefieldPupil(pixelSize, size, wavelength, NA, n)
    
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
                
    def GetPupil(self):
        u, v, R, pupil = self.basePupil.GeneratePupil(self.pixelSize, self.pupilSizeX, self.wavelength, self.NA, self.n)
        
    
    
        

    



def Plug(dsviewer):
    dsviewer.PSFTools = PSFTools(dsviewer)
    
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



