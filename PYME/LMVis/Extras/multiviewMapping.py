
##################
# multiviewMapping.py
#
# Copyright Andrew Barentine, David Baddeley
# david.baddeley@yale.edu
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
# AESB 08, 2016
##################

import wx
import numpy as np
from PYME.Analysis.points.DeClump import pyDeClump
import os
from PYME.IO.FileUtils import nameUtils
import json
import importlib

def foldX(pipeline):
    """

    At this point the origin of x should be the corner of the concatenated frame

    Args:
        pipeline:

    Returns: nothing
        Adds folded x-coordinates to the pipeline
        Adds channel assignments to the pipeline

    """
    xOG = pipeline.selectedDataSource.resultsSource.fitResults['fitResults']['x0']
    roiSizeNM = (pipeline.mdh['Multiview.ROISize'][1]*pipeline.mdh['voxelsize.x']*1000)  # voxelsize is in um
    xfold = xOG % roiSizeNM
    mvQuad = np.floor(xOG / roiSizeNM)

    pipeline.mapping.setMapping('xFolded', xfold)
    pipeline.mapping.setMapping('whichChan', mvQuad)
    #pipeline.mapping['xFolded'] = xfold
    #pipeline.mapping['whichChan'] = mvQuad
    return

def plotFolded(X, Y, multiviewChannels, title=''):
    """

    Args:
        X: array of localization x-positions
        Y: array of localization y-positions
        multiviewChannels: array of channel assignment of localizations
        title: title of plot

    Returns: nothing
        Plots unclumped raw localizations folded into the first channel

    """
    import matplotlib.pyplot as plt
    nChan = multiviewChannels.max()

    c = iter(plt.cm.rainbow(np.linspace(0, 1, nChan)))
    for ii in range(nChan):
        mask = (ii == multiviewChannels)
        plt.scatter(X[mask], Y[mask], c=next(c))
    plt.title(title)
    return

def plotRegistered(regX, regY, numChan, title=''):
    """

    Args:
        regX: list in which each element is an array of registered x-positions of molecules for a single channel
        regY: list in which each element is an array of registered y-positions of molecules for a single channel
        numChan: number of multiview channels
        title: title of plot

    Returns: nothing
        Plots molecules using the format that localization-clump positions are stored in.

    """
    import matplotlib.pyplot as plt
    plt.figure()
    c = iter(plt.cm.rainbow(np.linspace(0, 1, numChan)))
    for ii in range(numChan):
        plt.scatter(regX[ii], regY[ii], c=next(c), label='Chan #%i' % ii)
    plt.title(title)
    return

def pairMolecules(tIndex, x, y, whichChan, numChan, deltaX=[None], appearIn=np.arange(4)):
    """
    pairMolecules uses pyDeClump functions to group localization clumps into molecules for registration.

    Args:
        tIndex: from fitResults
        x: x positions of localizations AFTER having been folded into the first channel
        y: y positions of localizations
        whichChan: a vector containing channel assignments for each localization
        numChan: number of multiview channels
        deltaX: distance within which neighbors will be clumped is set by 2*deltaX[i])**2
        appearances: number of channels that must be present in a clump to be clumped

    Returns:
        x and y positions of molecules that were clumped, which channel those localizations are from,
        and which clump they were assigned to. Note that outputs are length = #molecules, and the keep
        vector that is return needs to be applied as: xkept = x[keep] in order to only look at kept molecules.
        Note that the returned x, y, tIndex and whichChan are resorted.

    """
    # sort everything in frame order
    I = tIndex.argsort()
    tIndex = tIndex[I]
    x = x[I]
    y = y[I]
    whichChan = whichChan[I]

    # group within a certain distance, potentially based on localization uncertainty
    if not deltaX[0]:
        deltaX = 100.*np.ones_like(x)
    # group localizations
    assigned = pyDeClump.findClumps(tIndex, x, y, deltaX)

    # only look at clumps with localizations from each channel
    clumps = np.unique(assigned)
    keptClumps = [np.array_equal(np.unique(whichChan[assigned == clumps[ii]]), appearIn) for ii in range(len(clumps))]
    #keptClumps = [(len(np.unique(whichChan[assigned == clumps[ii]])) >= appearances) for ii in range(len(clumps))]

    keptMoles = []
    # np.array_equal(clumps, np.arange(1, np.max(assigned) + 1)) evaluates to True
    for elem in assigned:
        keptMoles.append(elem in clumps[np.where(keptClumps)])
    keep = np.where(keptMoles)

    # don't clump molecules from the wrong channel (done by parsing modified whichChan to this function)
    ignoreChan = whichChan < 0
    numClump = np.max(assigned)
    igVec = np.arange(numClump, numClump + sum(ignoreChan))
    # give ignored channel localizations unique clump assignments
    assigned[ignoreChan] = igVec


    return x, y, whichChan, assigned, keep

def astigMAPism(pipeline, stigLib):
    """
    Look up table
    Args:
        sigVals:

    Returns:

    """
    import scipy.interpolate as terp
    whichChan = pipeline['whichChannel']
    # generate lookup table
    zVal = np.arange(stigLib['zRange'][0], stigLib['zRange'][1])

    sigCalX = {}  # np.zeros((len(zVal), numPlanes))
    sigCalY = {}  # np.zeros_like(sigCalX)

    # generate look up table of sorts
    for ii in np.unique(whichChan):
        #sigVals = stigLib['sigxTerp%i' % ii](zVal)
        #sigVals = stigLib['sigxTerp%i' % ii](zVal)
        # Zsigy = sigmaLibrary['sigyTerp%i' % ii](zVal)
        #astigLib['sigxTerp%i' % ii] = terp.UnivariateSpline(astigLib['PSF%i' % ii]['z'], astigLib['PSF%i' % ii]['sigmax'],
        #                                                    bbox=[lowerZ, upperZ])
        #astigLib['sigyTerp%i' % ii] = terp.UnivariateSpline(astigLib['PSF%i' % ii]['z'], astigLib['PSF%i' % ii]['sigmay'],
        #                                                    bbox=[lowerZ, upperZ])
        # FIXME: BBOX NEEDS DIFFERENT BOUNDS FOR EACH CHANNEL
        sigCalX['chan%i' % ii] = terp.UnivariateSpline(stigLib['PSF%i' % ii]['z'], stigLib['PSF%i' % ii]['sigmax'],
                                                            bbox=[stigLib['zRange'][0], stigLib['zRange'][1]])(zVal)
        sigCalY['chan%i' % ii] = terp.UnivariateSpline(stigLib['PSF%i' % ii]['z'], stigLib['PSF%i' % ii]['sigmay'],
                                                            bbox=[stigLib['zRange'][0], stigLib['zRange'][1]])(zVal)


class multiviewMapper:
    """

    multiviewMapper provides methods for registering multiview channels as acquired in multicolor or biplane imaging.
    Image frames for multiview data should have channels concatonated horizontally, such that the x-dimension is the
    only dimension that needs to be folded into the first channel.
    The shiftmaps are functions that interface like bivariate splines, but can be recreated from a dictionary of their
    fit-model parameters, which are stored in a dictionary in the shiftmap object.
    In the multiviewMapper class, shiftmaps for multiview data sources are stored in a dictionary of dictionaries.
    Each shiftmap is stored as a dictionary so that it can be easily written into a human-readable json file. These
    shiftmaps are then stored in the shiftWallet dictionary.

    """
    def __init__(self, visFr):
        self.visFr = visFr

        ID_CALIBRATE_SHIFTS = wx.NewId()
        visFr.extras_menu.Append(ID_CALIBRATE_SHIFTS, "Multiview - Calibrate Shifts")
        visFr.Bind(wx.EVT_MENU, self.OnCalibrateShifts, id=ID_CALIBRATE_SHIFTS)

        ID_MAP_XY = wx.NewId()
        visFr.extras_menu.Append(ID_MAP_XY, "Multiview - Map XY")
        visFr.Bind(wx.EVT_MENU, self.OnFoldAndMapXY, id=ID_MAP_XY)

        ID_MAP_Z = wx.NewId()
        visFr.extras_menu.Append(ID_MAP_Z, "Astigmatism - Map Z")
        visFr.Bind(wx.EVT_MENU, self.OnMapZ, id=ID_MAP_Z)
        return

    def applyShiftmaps_nonOrderConserving(self, x, y, numChan):
        """
        applyShiftmaps loads multiview shiftmap parameters from multiviewMapper.shiftWallet, reconstructs the shiftmap
        objects, applies them to the multiview data, and maps the positions registered to the first channel to the pipeline

        Args:
            x: vector of localization x-positions
            y: vector of localization y-positions
            numChan: number of multiview channels

        Returns: nothing
            Maps shifted x-, and y-positions into the pipeline
            xReg and yReg are both lists, where each element is an array of positions corresponding to a given channel

        """
        pipeline = self.visFr.pipeline

        # import shiftModel to be reconstructed
        model = self.shiftWallet['shiftModel'].split('.')[-1]
        shiftModule = importlib.import_module(self.shiftWallet['shiftModel'].split('.' + model)[0])
        shiftModel = getattr(shiftModule, model)

        # fixme: this needs to be done in such a way as to keep x and y matched up with the remaining fitResults
        xReg, yReg, chan = [x[0]], [y[0]], [np.zeros_like(x[0])]
        for ii in range(1, numChan):
            xReg.append(x[ii] + shiftModel(dict=self.shiftWallet['Chan0%s.X' % ii]).ev(x[ii], y[ii]))
            yReg.append(y[ii] + shiftModel(dict=self.shiftWallet['Chan0%s.Y' % ii]).ev(x[ii], y[ii]))
            chan.append(ii*np.ones_like(xReg[ii]))

        xReg = np.hstack(xReg)
        yReg = np.hstack(yReg)
        chan = np.hstack(chan)

        pipeline.mapping.setMapping('xReg', xReg)
        pipeline.mapping.setMapping('yReg', yReg)
        pipeline.mapping.setMapping('regChan', chan)
        #pipeline['xReg'] = xReg
        #pipeline['yReg'] = yReg
        return

    def applyShiftmaps(self, numChan):
        """
        applyShiftmaps loads multiview shiftmap parameters from multiviewMapper.shiftWallet, reconstructs the shiftmap
        objects, applies them to the multiview data, and maps the positions registered to the first channel to the pipeline

        Args:
            x: vector of localization x-positions
            y: vector of localization y-positions
            numChan: number of multiview channels

        Returns: nothing
            Maps shifted x-, and y-positions into the pipeline
            xReg and yReg are both lists, where each element is an array of positions corresponding to a given channel

        """
        pipeline = self.visFr.pipeline
        fres = pipeline.selectedDataSource.resultsSource.fitResults
        try:
            alreadyDone = pipeline.mapping.registered
            return
        except:
            pass

        # import shiftModel to be reconstructed
        model = self.shiftWallet['shiftModel'].split('.')[-1]
        shiftModule = importlib.import_module(self.shiftWallet['shiftModel'].split('.' + model)[0])
        shiftModel = getattr(shiftModule, model)


        x, y = pipeline.mapping.xFolded, fres['fitResults']['y0']
        chan = pipeline.mapping.whichChan
        # note that this will not throw out localizations outside of the frame, this will need to be done elsewhere
        for ii in range(1, numChan):
            chanMask = chan == ii
            x = x + chanMask*shiftModel(dict=self.shiftWallet['Chan0%s.X' % ii]).ev(x, y)
            y = y + chanMask*shiftModel(dict=self.shiftWallet['Chan0%s.Y' % ii]).ev(x, y)

        # replace x and y with shifted data
        #pipeline.mapping.resultsSource.resultsSource.fitResults['fitResults']['x0'] = x
        #pipeline.mapping.resultsSource.resultsSource.fitResults['fitResults']['y0'] = y

        # flag that this data has already been registered so it is not registered again
        pipeline.mapping.setMapping('registered', True)
        return x, y

    def OnFoldAndMapXY(self, event):
        """
        OnFoldAndMap uses shiftmaps stored in metadata (by default) or loaded through the GUI to register multiview channelss
        to the first channel.
        Args:
            event: GUI event

        Returns: nothing
            x- and y-positions will be registered to the first channel and stored in the pipeline dictionary as xReg and
            yReg. Their structure is described in applyShiftmaps

        """
        pipeline = self.visFr.pipeline
        fres = pipeline.selectedDataSource.resultsSource.fitResults

        try:  # load shiftmaps from metadata, if present
            self.shiftWallet = pipeline.mdh.__dict__['Shiftmap']
        except KeyError:
            try:  # load through GUI dialog
                fdialog = wx.FileDialog(None, 'Load shift field', wildcard='Shift Field file (*.sf)|*.sf',
                                        style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
                succ = fdialog.ShowModal()
                if (succ == wx.ID_OK):
                    fpath = fdialog.GetPath()
                    # load json
                    fid = open(fpath, 'r')
                    self.shiftWallet = json.load(fid)
                    fid.close()
            except:
                raise IOError('Shiftmaps not found in metadata and could not be loaded from file')

        numChan = pipeline.mdh['Multiview.NumROIs']
        # fold x-positions into the first channel
        foldX(pipeline)

        plotFolded(pipeline.mapping.xFolded, fres['fitResults']['y0'],
                            pipeline.mapping.whichChan, 'Raw')

        # organize x- and y-positions into list of arrays corresponding to channels
        #xfold, yfold = [], []
        #for ii in range(numChan):
        #    xfold.append(pipeline.mapping.xFolded[np.where(pipeline.mapping.whichChan == ii)])
        #    yfold.append(pipeline['y'][np.where(pipeline.mapping.whichChan == ii)])

        # apply shiftmaps
        x, y = self.applyShiftmaps(numChan)

        #plotRegistered(pipeline.mapping.xReg, pipeline.mapping.yReg,
        #                    numChan, 'After Registration')
        pipeline.mapping.setMapping('x', x)
        pipeline.mapping.setMapping('y', y)

    def OnCalibrateShifts(self, event):
        """

        OnRegisterMultiview generates multiview shiftmaps on bead-data. Only beads which show up in all channels are
        used to generate the shiftmap.

        Args:
            event: GUI event

        Returns: nothing
            Writes shiftmapWallet into metadata as well as saving a json formatted .sf file through a GUI dialog
        """
        from PYME.Analysis.points import twoColour
        pipeline = self.visFr.pipeline

        try:
            numChan = pipeline.mdh['Multiview.NumROIs']
        except KeyError:
            raise KeyError('You are either not looking at multiview Data, or your metadata is incomplete')

        # fold x position of channels into the first
        foldX(pipeline)

        # Match up molecules, note that all outputs are sorted in frame order!
        x, y, Chan, clumpID, keep = pairMolecules(pipeline['tIndex'], pipeline.mapping.xFolded, pipeline['fitResults_y0'],
                      pipeline.mapping.whichChan, numChan, appearIn=np.arange(numChan))  #, pipeline['error_x'])

        # only look at the ones which showed up in all channels
        x = x[keep], y = y[keep], Chan = Chan[keep], clumpID = clumpID[keep]
        # Generate raw shift vectors (map of displacements between channels) for each channel
        molList = np.unique(clumpID)
        numMoles = len(molList)

        dx = np.zeros((numChan - 1, numMoles))
        dy = np.zeros_like(dx)
        dxErr = np.zeros_like(dx)
        dyErr = np.zeros_like(dx)
        xClump, yClump, xStd, yStd = [], [], [], []
        self.shiftWallet = {}
        dxWallet, dyWallet = {}, {}
        for ii in range(numChan):
            chanMask = (Chan == ii)
            xChan = np.zeros(numMoles)
            yChan = np.zeros(numMoles)
            xChanStd = np.zeros(numMoles)
            yChanStd = np.zeros(numMoles)


            for ind in range(numMoles):
                # merge clumps within channels
                clumpMask = np.where(np.logical_and(chanMask, clumpID == molList[ind]))
                xChan[ind] = x[clumpMask].mean()
                yChan[ind] = y[clumpMask].mean()
                xChanStd[ind] = x[clumpMask].std()
                yChanStd[ind] = y[clumpMask].std()

            xClump.append(xChan)
            yClump.append(yChan)
            xStd.append(xChanStd)
            yStd.append(yChanStd)

            if ii > 0:
                dx[ii - 1, :] = xClump[0] - xClump[ii]
                dy[ii - 1, :] = yClump[0] - yClump[ii]
                dxErr[ii - 1, :] = np.sqrt(xStd[ii]**2 + xStd[0]**2)
                dyErr[ii - 1, :] = np.sqrt(yStd[ii]**2 + yStd[0]**2)
                # generate shiftmap between ii-th channel and the 0th channel
                dxx, dyy, spx, spy, good = twoColour.genShiftVectorFieldQ(xClump[0], yClump[0], dx[ii-1, :], dy[ii-1, :], dxErr[ii-1, :], dyErr[ii-1, :])
                # store shiftmaps in multiview shiftWallet
                self.shiftWallet['Chan0%s.X' % ii], self.shiftWallet['Chan0%s.Y' % ii] = spx.__dict__, spy.__dict__
                dxWallet['Chan0%s' % ii], dyWallet['Chan0%s' % ii] = dxx, dyy


        self.shiftWallet['shiftModel'] = '.'.join([spx.__class__.__module__, spx.__class__.__name__])
        # store shiftmaps in metadata
        pipeline.mdh.__dict__.__setitem__('Shiftmap', self.shiftWallet)
        # store shiftvectors in metadata
        pipeline.mdh.__dict__.__setitem__('chroma.dx', dxWallet)
        pipeline.mdh.__dict__.__setitem__('chroma.dy', dyWallet)
        # save shiftmaps
        defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + 'MultiView.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            fid = open(fpath, 'wb')
            json.dump(self.shiftWallet, fid)
            fid.close()

        # apply shiftmaps to clumped localizations
        self.applyShiftmaps(xClump, yClump, numChan)

        # organize x- and y-positions into list of arrays corresponding to channel
        xfold, yfold, tReg = [], [], []
        for ii in range(numChan):
            xfold.append(pipeline.mapping.xFolded[np.where(pipeline.mapping.whichChan == ii)])
            yfold.append(pipeline['fitResults_y0'][np.where(pipeline.mapping.whichChan == ii)])
            tReg.append(pipeline['tIndex'][np.where(pipeline.mapping.whichChan == ii)])

        pipeline.mapping.setMapping('tIndex_reg', np.hstack(tReg))

        plotRegistered(xfold, yfold, numChan, 'Raw')

        plotRegistered(xClump, yClump, numChan, 'Clumped')

        plotRegistered(pipeline.mapping.xReg, pipeline.mapping.yReg,
                            numChan, 'After Registration')


    def OnMapZ(self, event):
        pipeline = self.visFr.pipeline
        fres = pipeline.selectedDataSource.resultsSource.fitResults

        try:
            numChan = pipeline.mdh['Multiview.NumROIs']
            chanColor = [0, 0, 1, 1]  # FIXME: pipeline.mdh['Multiview.ColorInfo']
            numPlanes = numChan / len(np.unique(chanColor))
            chanPlane = [0, 1, 0, 1]  # FIXME: make this automatic np.unique(chanColor, return_index=True)
        except AttributeError:
            numChan = 1
            chanColor = [0]
            numPlanes = 1
        try:  # load astigmatism calibrations from metadata, if present
            stigLib = pipeline.mdh['astigLib']
        except AttributeError:
            try:  # load through GUI dialog
                fdialog = wx.FileDialog(None, 'Load Astigmatism Calibration', #wildcard='Shift Field file (*.sf)|*.sf',
                                        style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
                succ = fdialog.ShowModal()
                if (succ == wx.ID_OK):
                    fpath = fdialog.GetPath()
                    # load json
                    fid = open(fpath, 'r')
                    stigLib = json.load(fid)
                    fid.close()
            except:
                raise IOError('Astigmatism sigma-Z mapping information not found')

        # make sure xy-registration has already happened:
        if 'registered' not in pipeline.keys():
            print('registering multiview channels in x-y plane')
            self.OnFoldAndMapXY(event)

        # clump molecules
        pairs, clump, xco, yco, xcoErr, ycoErr, xsig, ysig, clist = [], [], [], [], [], [], [], [], []
        # sort error and sigmas as x, y, and channel will be sorted
        #I = pipeline['tIndex'].argsort()
        #errX = pipeline['error_x'][I]
        #errY = pipeline['error_y'][I]
        #xs = pipeline['fitResults_sigmax'][I]
        #ys = pipeline['fitResults_sigmay'][I]
        #whichChan = pipeline.mapping.whichChan[I]
        #pairs = np.zeros_like(errX, dtype=bool)
        fresCopy = np.copy(fres)
        # inject xReg and yReg
        #fitResCopy['fitResults']['x0'] = pipeline.mapping.x
        #fitResCopy['fitResults']['y0'] = pipeline.mapping.y
        #fres['fitResults'].setfield(2.0, [('test', '<i4')])
        #fitResCopy.setfield(pipeline.mapping.whichChan, ['whichChan', '<i4'])
        # fitResCopy.whichChan = np.copy(pipeline.mapping.whichChan)
        # want to hack coalesceClumps to shrink whichChan as well as average sigma values separately for each plane
        dt = list(fres.dtype.descr)
        addDT = []
        addDT += [('sigmax_Plane%i' % pi, '<f4') for pi in range(numPlanes)]
        addDT += [('sigmay_Plane%i' % pi, '<f4') for pi in range(numPlanes)]
        dt[1] = list(dt[1])
        dt[2] = list(dt[2])
        #dt1, dt2 = list(dt[1]), list(dt[2])
        dt[1][1] += addDT
        dt[2][1] += addDT
        #dt[1][1] = dt1
        #dt[2][1] = dt2
        dt = dt + [('whichChannel', '<i4')]
        dt[1], dt[2] = tuple(dt[1]), tuple(dt[2])
        fresCopy = np.empty_like(fres, dtype=dt)
        # copy over existing fitresults:
        for fr in fres.dtype.descr:
            fresCopy[fr[0]] = fres[fr[0]]
        fresCopy['whichChannel'] = np.int32(pipeline.mapping.whichChan)
        for pind in range(numPlanes):
            pMask = [(chanPlane[fresCopy['whichChannel'][p]] == pind) for p in range(len(fresCopy['whichChannel']))]
            fresCopy['fitResults']['sigmax_Plane%i' % pind] = pMask*fresCopy['fitResults']['sigmax']
            fresCopy['fitResults']['sigmay_Plane%i' % pind] = pMask*fresCopy['fitResults']['sigmay']
            fresCopy['fitError']['sigmax_Plane%i' % pind] = pMask*fresCopy['fitError']['sigmax']
            fresCopy['fitError']['sigmay_Plane%i' % pind] = pMask*fresCopy['fitError']['sigmay']
            # replace zeros in fiterror with infs so their weights are zero
            fresCopy['fitError']['sigmax_Plane%i' % pind][fresCopy['fitError']['sigmax_Plane%i' % pind] == 0] = np.inf
            fresCopy['fitError']['sigmay_Plane%i' % pind][fresCopy['fitError']['sigmay_Plane%i' % pind] == 0] = np.inf
        for cind in range(len(chanColor)):
            # trick pairMolecules function by tweaking the channel vector
            # this needs to be unsorted at this point
            planeInColorChan = np.copy(fresCopy['whichChannel']) #fixme: needs to shrink with fitResCopy
            chanColor = np.array(chanColor)
            ignoreChans = np.where(chanColor != cind)[0]
            igMask = [mm in ignoreChans.tolist() for mm in planeInColorChan]
            planeInColorChan[np.where(igMask)] = -9
            # clumpID is assigned, paired is keep
            x, y, Chan, clumpID, paired = pairMolecules(fresCopy['tIndex'], fresCopy['fitResults']['x0'], fresCopy['fitResults']['y0'],
                          planeInColorChan, numChan, deltaX=100*fresCopy['fitError']['x0'],
                                                        appearIn=np.hstack([-9, np.where(chanColor == cind)[0]]))

            #clumpedRes['fitResults'] = clumpedRes['fitResults'][np.where(planeInColorChan >= 0)[0]]
            # FIXME: fix coalesceClumps to average sigmas separately for each plane in channel
            # numClumps = len(clumpID)
            clumpVec = np.unique(clumpID)
            for ci in range(len(clumpVec)):
                cMask = clumpID == clumpVec[ci]
                clumpID[cMask] = ci
            clumpedRes = pyDeClump.coalesceClumps(fresCopy, clumpID)
            fresCopy = clumpedRes

        # create data source for our clumped dat
        from PYME.LMVis.inpFilt import fitResultsSource
        pipeline.addDataSource('Clumped', fitResultsSource(clumpedRes))
        pipeline.selectDataSource('Clumped')
        self.visFr.RegenFilter()
        self.visFr.CreateFoldPanel()

        z = astigMAPism(pipeline, stigLib)



def Plug(visFr):
    """Plugs this module into the gui"""
    multiviewMapper(visFr)
