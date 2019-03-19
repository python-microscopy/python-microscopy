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
import wx.html2
import wx.grid

from jinja2 import Environment, PackageLoader
env = Environment(loader=PackageLoader('PYME.DSView.modules', 'templates'))
#import pylab
#from PYME.IO.image import ImageStack
try:
    from enthought.traits.api import HasTraits, Float, Int, Bool
    from enthought.traits.ui.api import View, Item
    from enthought.traits.ui.menu import OKButton
except ImportError:
    from traits.api import HasTraits, Float, Int, Bool
    from traitsui.api import View, Item
    from traitsui.menu import OKButton

from graphViewPanel import *
from PYME.Analysis.PSFEst import psfQuality

def remove_newlines(s):
    s = '<>'.join(s.split('\n\n'))
    s = ' '.join(s.split())
    return '\n'.join(s.split('<>'))

def find_and_add_zRange(astig_library, rough_knot_spacing=50.):
    """
    Find range about highest intensity point over which sigmax - sigmay is monotonic.
    Note that astig_library[psfIndex]['zCenter'] should contain the offset in nm to the brightest z-slice

    Parameters
    ----------
    astig_library : List
        Elements are dictionaries containing PSF fit information
    rough_knot_spacing : Float
        Smoothing is applied to (sigmax-sigmay) before finding the region over which it is monotonic. A cubic spline is
        fit to (sigmax-sigmay) using knots spaced roughly be rough_knot_spacing (units of nanometers, i.e. that of
        astig_library[ind]['z']). To make deciding the knots convenient, they are spaced an integer number of z-steps,
        so the actual knot spacing is rounded to this.

    Returns
    -------
    astig_library : List
        The astigmatism calibration list which is taken as an input is modified in place and returned.

    """
    #TODO - move this to an astigmatism calibration module. This doesn't belong here.
    
    import scipy.interpolate as terp
    # from scipy.stats import mode
    for ii in range(len(astig_library)):
        # figure out where to place knots. Note that we subsample our z-positions so we satisfy Schoenberg-Whitney
        # conditions, i.e. that our spline has adequate support
        z_steps = np.unique(astig_library[ii]['z'])
        dz_med = np.median(np.diff(z_steps))
        smoothing_factor = int(rough_knot_spacing / dz_med)
        knots = z_steps[1:-1:smoothing_factor]
        # make the spline
        dsig = terp.LSQUnivariateSpline(astig_library[ii]['z'], astig_library[ii]['dsigma'], knots)

        # mask where the sign is the same as the center
        zvec = np.linspace(np.min(astig_library[ii]['z']), np.max(astig_library[ii]['z']), 1000)
        sgn = np.sign(np.diff(dsig(zvec)))
        halfway = np.absolute(zvec - astig_library[ii]['zCenter']).argmin()  # len(sgn)/2
        notmask = sgn != sgn[halfway]

        # find region of dsigma which is monotonic after smoothing
        try:
            lowerZ = zvec[np.where(notmask[:halfway])[0].max()]
        except ValueError:
            lowerZ = zvec[0]
        try:
            upperZ = zvec[(halfway + np.where(notmask[halfway:])[0].min() - 1)]
        except ValueError:
            upperZ = zvec[-1]
        astig_library[ii]['zRange'] = [lowerZ, upperZ]

    return astig_library

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
        from PYME.Analysis.PSFGen import fourierHNA
        #print 'b'
        import numpy as np
        try:
            d = self.image.data[:,:,:,0].squeeze()
            I = d[:,:,d.shape[2]/2].sum()

            vs = 1e3*np.array([self.image.mdh['voxelsize.x'], self.image.mdh['voxelsize.y'],self.image.mdh['voxelsize.z']])

            #print 'fi'
            FI = cramerRao.CalcFisherInformZn2(d*(2e3/I) + self.background, 100, voxelsize=vs)
            #print 'crb'
            self.crb = cramerRao.CalcCramerReoZ(FI)
            #print 'crbd'

            z_ = np.arange(d.shape[2])*self.image.mdh['voxelsize.z']*1.0e3
            self.z_ = z_ - z_.mean()

            ps_as = fourierHNA.GenAstigPSF(self.z_, dx=vs[0], strength=2)
            I = ps_as[:,:,ps_as.shape[2]/2].sum()
            self.crb_as = (cramerRao.CalcCramerReoZ(cramerRao.CalcFisherInformZn2(ps_as*2000/I + self.background, 500, voxelsize=vs)))

            self.draw()
        except np.linalg.linalg.LinAlgError:
            # don't hang if we can't compute the CRLB
            import traceback
            traceback.print_exc()



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
    intermediateUpdates = Bool(False)
    
    view = View(Item('wavelength'),
                Item('NA'),
                Item('pupilSize'),
                Item('iterations'), 
                Item('intermediateUpdates'),
                buttons=[OKButton])
    
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        
        dsviewer.AddMenuItem('Processing', "Extract &Pupil Function", self.OnExtractPupil)
        dsviewer.AddMenuItem('Processing', "Cramer-Rao Bound vs Background ", self.OnCalcCRB3DvsBG)
        dsviewer.AddMenuItem('Processing', "PSF Background Correction", self.OnSubtractBackground)
        dsviewer.AddMenuItem('Processing', "Perform Astigmatic Calibration", self.OnCalibrateAstigmatism)
        #wx.EVT_MENU(dsviewer, PROC_LABEL, self.OnLabel)

    def OnExtractPupil(self, event):
        import numpy as np
        import pylab
        from PYME.Analysis.PSFGen import fourierHNA
        
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        z_ = np.arange(self.image.data.shape[2])*self.image.mdh['voxelsize.z']*1.e3
        z_ -= z_.mean()  
        
        self.configure_traits(kind='modal')
        
        #pupil = fourierHNA.ExtractPupil(np.maximum(self.image.data[:,:,:] - .001, 0), z_, self.image.mdh['voxelsize.x']*1e3, self.wavelength, self.NA, nIters=self.iterations, size=self.pupilSize)

        pupil = fourierHNA.ExtractPupil(self.image.data[:,:,:], z_, self.image.mdh['voxelsize.x']*1e3, self.wavelength, self.NA, nIters=self.iterations, size=self.pupilSize, intermediateUpdates=self.intermediateUpdates)
                
        
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

    def OnCalibrateAstigmatism(self, event):
        #TODO - move all non-GUI logic for this out of this file?
        from PYME.recipes.measurement import FitPoints
        from PYME.IO.FileUtils import nameUtils
        import matplotlib.pyplot as plt
        import mpld3
        import json
        from PYME.Analysis.PSFEst import extractImages
        import wx

        # query user for type of calibration
        # NB - GPU fit is not enabled here because it exits on number of iterations, which is not necessarily convergence for very bright beads!
        ftypes = ['BeadConvolvedAstigGaussFit', 'AstigGaussFitFR']  # , 'AstigGaussGPUFitFR']
        fitType_dlg = wx.SingleChoiceDialog(self.dsviewer, 'Fit-type selection', 'Fit-type selection', ftypes)
        fitType_dlg.ShowModal()
        fitMod = ftypes[fitType_dlg.GetSelection()]

        if (fitMod == 'BeadConvolvedAstigGaussFit') and ('Bead.Diameter' not in self.image.mdh.keys()):
            beadDiam_dlg = wx.NumberEntryDialog(None, 'Bead diameter in nm', 'diameter [nm]', 'diameter [nm]', 100, 1, 9e9)
            beadDiam_dlg.ShowModal()
            beadDiam = float(beadDiam_dlg.GetValue())
            # store this in metadata
            self.image.mdh['Analysis.Bead.Diameter'] = beadDiam



        ps = self.image.pixelSize

        obj_positions = {}

        obj_positions['x'] = ps*self.image.data.shape[0]*0.5*np.ones(self.image.data.shape[2])
        obj_positions['y'] = ps * self.image.data.shape[1] * 0.5 * np.ones(self.image.data.shape[2])
        obj_positions['t'] = np.arange(self.image.data.shape[2])
        z = np.arange(self.image.data.shape[2]) * self.image.mdh['voxelsize.z'] * 1.e3
        obj_positions['z'] = z - z.mean()

        ptFitter = FitPoints()
        ptFitter.trait_set(roiHalfSize=11)
        ptFitter.trait_set(fitModule=fitMod)

        namespace = {'input' : self.image, 'objPositions' : obj_positions}

        results = []

        for chanNum in range(self.image.data.shape[3]):
            # get z centers
            dx, dy, dz = extractImages.getIntCenter(self.image.data[:, :, :, chanNum])

            ptFitter.trait_set(channel=chanNum)
            ptFitter.execute(namespace)

            res = namespace['fitResults']

            dsigma = abs(res['fitResults_sigmax']) - abs(res['fitResults_sigmay'])
            valid = ((res['fitError_sigmax'] > 0) * (res['fitError_sigmax'] < 50)* (res['fitError_sigmay'] < 50)*(res['fitResults_A'] > 0) > 0)

            results.append({'sigmax': abs(res['fitResults_sigmax'][valid]).tolist(),'error_sigmax': abs(res['fitError_sigmax'][valid]).tolist(),
                            'sigmay': abs(res['fitResults_sigmay'][valid]).tolist(), 'error_sigmay': abs(res['fitError_sigmay'][valid]).tolist(),
                            'dsigma': dsigma[valid].tolist(), 'z': obj_positions['z'][valid].tolist(), 'zCenter': obj_positions['z'][int(dz)]})

        #generate new tab to show results
        use_web_view = True
        if not '_astig_view' in dir(self):
            try:
                self._astig_view= wx.html2.WebView.New(self.dsviewer)
                self.dsviewer.AddPage(self._astig_view, True, 'Astigmatic calibration')

            except NotImplementedError:
                use_web_view = False

        # find reasonable z range for each channel, inject 'zRange' into the results. FIXME - injection is bad
        results = find_and_add_zRange(results)

        #do plotting
        plt.ioff()
        f = plt.figure(figsize=(10, 4))

        colors = iter(plt.cm.Dark2(np.linspace(0, 1, 2*self.image.data.shape[3])))
        plt.subplot(121)
        for i, res in enumerate(results):
            nextColor1 = next(colors)
            nextColor2 = next(colors)
            lbz = np.absolute(res['z'] - res['zRange'][0]).argmin()
            ubz = np.absolute(res['z'] - res['zRange'][1]).argmin()
            plt.plot(res['z'], res['sigmax'], ':', c=nextColor1)  # , label='x - %d' % i)
            plt.plot(res['z'], res['sigmay'], ':', c=nextColor2)  # , label='y - %d' % i)
            plt.plot(res['z'][lbz:ubz], res['sigmax'][lbz:ubz], label='x - %d' % i, c=nextColor1)
            plt.plot(res['z'][lbz:ubz], res['sigmay'][lbz:ubz], label='y - %d' % i, c=nextColor2)

        #plt.ylim(-200, 400)
        plt.grid()
        plt.xlabel('z position [nm]')
        plt.ylabel('Sigma [nm]')
        plt.legend()

        plt.subplot(122)
        colors = iter(plt.cm.Dark2(np.linspace(0, 1, self.image.data.shape[3])))
        for i, res in enumerate(results):
            nextColor = next(colors)
            lbz = np.absolute(res['z'] - res['zRange'][0]).argmin()
            ubz = np.absolute(res['z'] - res['zRange'][1]).argmin()
            plt.plot(res['z'], res['dsigma'], ':', lw=2, c=nextColor)  # , label='Chan %d' % i)
            plt.plot(res['z'][lbz:ubz], res['dsigma'][lbz:ubz], lw=2, label='Chan %d' % i, c=nextColor)
        plt.grid()
        plt.xlabel('z position [nm]')
        plt.ylabel('Sigma x - Sigma y [nm]')
        plt.legend()

        plt.tight_layout()

        plt.ion()
        #dat = {'z' : objPositions['z'][valid].tolist(), 'sigmax' : res['fitResults_sigmax'][valid].tolist(),
        #                   'sigmay' : res['fitResults_sigmay'][valid].tolist(), 'dsigma' : dsigma[valid].tolist()}


        if use_web_view:
            fig = mpld3.fig_to_html(f)
            data = json.dumps(results)

            template = env.get_template('astigCal.html')
            html = template.render(astigplot=fig, data=data)
            #print html
            self._astig_view.SetPage(html, '')
        else:
            plt.show()

        fdialog = wx.FileDialog(None, 'Save Astigmatism Calibration as ...',
            wildcard='Astigmatism Map (*.am)|*.am', style=wx.SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath())  #, defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            fid = open(fpath, 'wb')
            json.dump(results, fid, indent=4, sort_keys=True)
            fid.close()


        return results

        
        
    def OnSubtractBackground(self, event):
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        from PYME.Analysis.PSFEst import extractImages

        d_bg = extractImages.backgroundCorrectPSFWF(self.image.data[:,:,:])
        
        
        
        im = ImageStack(d_bg, titleStub = 'Filtered Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        
        dv = ViewIm3D(im, mode='psf', glCanvas=self.dsviewer.glCanvas)

        
    def OnCalcCRB(self, event):
        #print 'f'
        from PYME.Analysis import cramerRao
        from PYME.Analysis.PSFGen import fourierHNA
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
        from PYME.Analysis.PSFGen import fourierHNA
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



