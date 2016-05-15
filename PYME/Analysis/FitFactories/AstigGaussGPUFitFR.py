    #!/usr/bin/python

##################
# AstigGaussGPUFitFR.py
#
# This FitFactory calls the PYMEwarpDrive package to perform
# candidate molecule detection and fitting on the GPU, which
# takes into account an sCMOS specific noise model. This
# GPU package is available on-request for academic use. If
# interested, please contact David Baddeley or Joerg Bewersdorf.
# The fitting itself is described in DOI: 10.1038/nmeth.2488
#
# Copyright David Baddeley, 2009
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
# AESB 04,2016: modified GaussMultiFitSR to work with PYMEwarpDrive
##################


from PYME.Analysis._fithelpers import *

try:
    from warpDrive import *
except ImportError:
    print("GPU fitting available on-request for academic use. Please contact David Baddeley or Joerg Bewersdorf.")

##################
# Model Function, only for reference in this case.
def f_gaussAstigSlow(p, X, Y):
    """2D Gaussian model function with independent sigx, sigy - parameter vector [A, x0, y0, sx, sy]"""
    #x0, y0, A, sx, sy, c, = p
    x0, y0, A, sx, sy = p
    return A*np.exp(-(X[:,None]-x0)**2/(2*sy**2) - (Y[None,:] - y0)**2/(2*sx**2))  # + c
#####################


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('y0', '<f4'), ('x0', '<f4'), #Fang and Davids xys are swapped
                              ('photons', '<f4'),
                              ('background', '<f4'),
                              ('sigmay', '<f4'),('sigmax', '<f4')]),

              ('fitError', [('y0', '<f4'), ('x0', '<f4'),
                              ('photons', '<f4'),
                              ('background', '<f4'),
                              ('sigmay', '<f4'),('sigmax', '<f4')]),
              ('resultCode', '<i4'),
              ('LLH', '<f4'),
              ('nFit', '<i4')]


def GaussianFitResultR(fitResults, metadata, resultCode=-1, fitErr=None, LLH=None, nEvents=1):
    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')
        LLH = np.zeros(nEvents)

    tIndex = metadata.getOrDefault('tIndex', 0)

    return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, LLH, nEvents)], dtype=fresultdtype)

_warpDrive = None

class GaussianFitFactory:
    X = None
    Y = None
    
    def __init__(self, data, metadata, fitfcn=None, background=None, noiseSigma=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. '''

        self.data = np.ascontiguousarray(np.squeeze(data), dtype=np.float32)
        self.metadata = metadata

        ''' next 3 lines are currently unused '''
        self.background = background
        self.noiseSigma = noiseSigma
        self.fitfcn = fitfcn




    def FindAndFit(self, threshold=4, gui=False, cameraMaps=None):
        global _warpDrive  # One warpDrive instance for each taskWorker instance

        # get varmap and flatmap
        varmap = cameraMaps.getVarianceMap(self.metadata)
        if not np.isscalar(varmap):
            self.varmap = np.ascontiguousarray(varmap)
        else:
            self.varmap = varmap*np.ones_like(self.data)

        #fixme currently detector object takes gain, flatmap is one over this. need to switch var maps of our cameras over eventually and reconcile
        flatmap = cameraMaps.getFlatfieldMap(self.metadata)
        if not np.isscalar(flatmap):
            self.flatmap = np.ascontiguousarray(1./flatmap)
        else:
            #flatmap = self.metadata['Camera.TrueEMGain']*self.metadata['Camera.ElectronsPerCount']
            self.flatmap = (1./flatmap)*np.ones_like(self.data)

        # subtract darkmap
        darkmap = cameraMaps.getDarkMap(self.metadata)
        self.data -= darkmap #same op for scalar or array

        # Account for any changes we need to make in memory allocation on the GPU
        if not _warpDrive:
            #Initialize new detector object for this CPU thread, we're going plaid
            dfilter1 = normUnifFilter(12)
            dfilter2 = normUnifFilter(6)
            _warpDrive = detector(np.shape(self.data), self.data.dtype.itemsize, dfilter1, dfilter2)
            _warpDrive.allocateMem()
            _warpDrive.prepvar(self.varmap, self.flatmap)
            ''' If the data is coming from a different region of the camera, reallocate
            note that 'and' is short circuiting in Python. Just check the first 20x20 elements'''
        elif _warpDrive.data.shape == self.data.shape:
            if (not np.array_equal(self.varmap[:20, :20], _warpDrive.varmap[:20, :20])):
                _warpDrive.prepvar(self.varmap, self.flatmap)
        else:  # we know that we need to allocate and prepvar
            _warpDrive.allocateMem()
            _warpDrive.prepvar(self.varmap, self.varmap)

        #PYME ROISize is a half size
        roiSize = int(2*self.metadata.getEntry('Analysis.ROISize') + 1)

        #######################
        # Actually do the fits
        _warpDrive.smoothFrame(self.data)
        _warpDrive.getCand(threshold, roiSize)
        if _warpDrive.candCount == 0:
            resList = np.empty(0, FitResultsDType)
            return resList
        _warpDrive.fitItSlow(roiSize)

        # LLH: (N); dpars and CRLB (N, 6)
        #convert pixels to nm; voxelsize in units of um
        _warpDrive.dpars[:, 0] *= (1000*self.metadata.voxelsize.y)
        _warpDrive.dpars[:, 1] *= (1000*self.metadata.voxelsize.x)
        _warpDrive.dpars[:, 4] *= (1000*self.metadata.voxelsize.y)
        _warpDrive.dpars[:, 5] *= (1000*self.metadata.voxelsize.x)

        fitErrors=None
        LLH = None


        if _warpDrive.calcCRLB:
            LLH = _warpDrive.LLH
            # fixme: Should never have negative CRLB, yet Yu reports ocassional instances in Matlab verison, check
            _warpDrive.CRLB[:, 0] = np.sqrt(np.abs(_warpDrive.CRLB[:, 0]))*(1000*self.metadata.voxelsize.y)
            _warpDrive.CRLB[:, 1] = np.sqrt(np.abs(_warpDrive.CRLB[:, 1]))*(1000*self.metadata.voxelsize.x)
            _warpDrive.CRLB[:, 4] = np.sqrt(np.abs(_warpDrive.CRLB[:, 4]))*(1000*self.metadata.voxelsize.y)
            _warpDrive.CRLB[:, 5] = np.sqrt(np.abs(_warpDrive.CRLB[:, 5]))*(1000*self.metadata.voxelsize.x)
            fitErrors = _warpDrive.CRLB

        #return self.chan1.dpars # each fit produces column vector of results, append them all horizontally for return
        resList = np.empty(_warpDrive.candCount, FitResultsDType)
        resultCode = 0

        # package our results with the right labels
        for ii in range(_warpDrive.candCount):
            resList[ii] = GaussianFitResultR(_warpDrive.dpars[ii, :], self.metadata, resultCode, fitErrors[ii, :], LLH[ii], _warpDrive.candCount)

        return np.hstack(resList)


    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        X = 1e3*md.voxelsize.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = 1e3*md.voxelsize.y*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_gaussAstigSlow(params, X, Y), X[0], Y[0], 0)

# so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

#this means that factory is reponsible for it's own object finding and implements
#a GetAllResults method that returns a list of localisations
MULTIFIT=True

import PYME.Analysis.MetaDataEdit as mde

PARAMETERS = [#mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              #mde.FilenameParam('PSFFilename', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
              #mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              mde.FloatParam('Analysis.ROISize', u'ROI half size', 7.5),
              #mde.FloatParam('Analysis.ResidualMax', 'Max residual:', 0.25),
              #mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              #mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'y', choices=['x', 'y'])
              ]
              
DESCRIPTION = 'Astigmatic Gaussian fitting performed at warp-speed on the GPU'
LONG_DESCRIPTION = 'Astigmatic Gaussian fitting on the GPU: Fits astigmatic gaussian with sCMOS noise model. Uses it\'s own object detection routine'
USE_FOR = '2D Astigmatism with sCMOS noise model'