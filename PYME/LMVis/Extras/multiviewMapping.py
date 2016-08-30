
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
import scipy.interpolate as terp
from PYME.LMVis.inpFilt import cachingResultsFilter  # mappingFilter  # fitResultsSource


def foldX(pipeline):
    """

    At this point the origin of x should be the corner of the concatenated frame

    Args:
        pipeline:

    Returns: nothing
        Adds folded x-coordinates to the pipeline
        Adds channel assignments to the pipeline

    """
    roiSizeNM = (pipeline.mdh['Multiview.ROISize'][1]*pipeline.mdh['voxelsize.x']*1000)  # voxelsize is in um

    pipeline.selectedDataSource.addVariable('roiSizeNM', roiSizeNM)
    pipeline.selectedDataSource.addVariable('numChannels', len(pipeline.mdh['Multiview.ChannelColor']) - 1)

    pipeline.addColumn('chromadx', 0*pipeline['x'])
    pipeline.addColumn('chromady', 0*pipeline['y'])

    pipeline.selectedDataSource.setMapping('whichChan', 'clip(floor(x/roiSizeNM), 0, numChannels).astype(int)')
    pipeline.selectedDataSource.setMapping('x', 'x%roiSizeNM + chromadx')
    pipeline.selectedDataSource.setMapping('y', 'y + chromady')

    return

def plotFolded(X, Y, multiviewChannels, title=''):
    """
    Plots localizations with color based on multiview channel
    Args:
        X: array of localization x-positions
        Y: array of localization y-positions
        multiviewChannels: array of channel assignment of localizations
        title: title of plot

    Returns: nothing

    """
    import matplotlib.pyplot as plt
    plt.figure()
    nChan = len(np.unique(multiviewChannels))

    c = iter(plt.cm.rainbow(np.linspace(0, 1, nChan)))
    for ii in range(nChan):
        mask = (ii == multiviewChannels)
        plt.scatter(X[mask], Y[mask], c=next(c), label='Chan #%i' % ii)
    plt.title(title)
    plt.legend()
    return

def pairMolecules(tIndex, x, y, whichChan, deltaX=[None], appearIn=np.arange(4), nFrameSep=5):
    """
    pairMolecules uses pyDeClump functions to group localization clumps into molecules for registration.

    Args:
        tIndex: from fitResults
        x: x positions of localizations AFTER having been folded into the first channel
        y: y positions of localizations
        whichChan: a vector containing channel assignments for each localization
        deltaX: distance within which neighbors will be clumped is set by 2*deltaX[i])**2
        appearIn: a clump must have localizations in each of these channels in order to be a keep-clump
        nFrameSep: number of frames a molecule is allowed to blink off and still be clumped as the same molecule

    Returns:
        assigned: clump assignments for each localization. Note that molecules whose whichChan entry is set to a
            negative value will not be clumped, i.e. they will have a unique value in assigned.
        keep: a boolean vector encoding which molecules are in kept clumps
        Note that outputs are of length #molecules, and the keep vector that is returned needs to be applied
        as: xkept = x[keep] in order to only look at kept molecules.

    """
    # group within a certain distance, potentially based on localization uncertainty
    if not deltaX[0]:
        deltaX = 100.*np.ones_like(x)
    # group localizations
    assigned = pyDeClump.findClumps(tIndex, x, y, deltaX, nFrameSep)
    # print assigned.min()
    # only look at clumps with localizations from each channel
    clumps = np.unique(assigned)
    # Note that this will never be a keep clump if an ignore channel is present...
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
    igVec = np.arange(numClump + 1, numClump + 1 + sum(ignoreChan))
    # give ignored channel localizations unique clump assignments
    assigned[ignoreChan] = igVec


    return assigned, keep

def applyShiftmaps(pipeline, shiftWallet, numChan):
    """
    applyShiftmaps loads multiview shiftmap parameters from multiviewMapper.shiftWallet, reconstructs the shiftmap
    objects, applies them to the multiview data, and maps the positions registered to the first channel to the pipeline

    Args:
        x: vector of localization x-positions
        y: vector of localization y-positions
        numChan: number of multiview channels

    Returns: nothing
        Adds shifts into the pipeline which will then be applied automatically by the mappingFilter (see foldX)

    """

    model = shiftWallet['shiftModel'].split('.')[-1]
    shiftModule = importlib.import_module(shiftWallet['shiftModel'].split('.' + model)[0])
    shiftModel = getattr(shiftModule, model)


    x, y = pipeline.mapping['x'], pipeline.mapping['y']

    # FIXME: the camera roi positions below would not account for the multiview data source
    #x = x + pipeline.mdh['Camera.ROIX0']*pipeline.mdh['voxelsize.x']*1.0e3
    #y = y + pipeline.mdh['Camera.ROIY0']*pipeline.mdh['voxelsize.y']*1.0e3
    chan = pipeline.mapping['whichChan']

    dx = 0
    dy = 0
    for ii in range(1, numChan):
        chanMask = chan == ii
        dx = dx + chanMask*shiftModel(dict=shiftWallet['Chan0%s.X' % ii]).ev(x, y)
        dy = dy + chanMask*shiftModel(dict=shiftWallet['Chan0%s.Y' % ii]).ev(x, y)

    pipeline.addColumn('chromadx', dx)
    pipeline.addColumn('chromady', dy)

def astigMAPism(fres, stigLib, chanPlane):
    """
    Generates a look-up table of sorts for z based on sigma x/y fit results and calibration information. If a molecule
    appears on multiple planes, sigma values from both planes will be used in the look up.
    Args:
        fres: dictionary-like object containing relevant fit results
        stigLib: library of astigmatism calibration dictionaries corresponding to each multiview channel, which are
            used to recreate shiftmap objects
        chanPlane: list of which plane each channel corresponds to, e.g. [0, 0, 1, 1]

    Returns:
        z: an array of z-positions for each molecule in nm (assuming proper units were used in astigmatism calibration)

    """
    # fres = pipeline.selectedDataSource.resultsSource.fitResults
    numMols = len(fres['fitResults_x0'])
    whichChan = np.array(fres['whichChan'], dtype=np.int32)
    # stigLib['zRange'] contains the extrema of acceptable z-positions looking over all channels

    # find overall min and max z values
    zrange = np.nan*np.ones(2)
    for zi in range(len(stigLib)):
        zrange = [np.nanmin([stigLib[zi]['zRange'][0], zrange[0]]), np.nanmax([stigLib[zi]['zRange'][1], zrange[1]])]
    # generate z vector for interpolation
    zVal = np.arange(zrange[0], zrange[1])

    sigCalX = {}
    sigCalY = {}

    z = np.zeros(numMols)

    # generate look up table of sorts
    for ii in np.unique(whichChan):
        zdat = np.array(stigLib[ii]['z'])
        # find indices of range we trust
        zrange = stigLib[ii]['zRange']
        lowsubZ , upsubZ = np.absolute(zdat - zrange[0]), np.absolute(zdat - zrange[1])
        lowZLoc = np.argmin(lowsubZ)
        upZLoc = np.argmin(upsubZ)

        sigCalX['chan%i' % ii] = terp.UnivariateSpline(zdat[lowZLoc:upZLoc],
                                                       np.array(stigLib[ii]['sigmax'])[lowZLoc:upZLoc],
                                                       ext='zeros')(zVal)
                                                            # bbox=stigLib['PSF%i' % ii]['zrange'], ext='zeros')(zVal)
        sigCalY['chan%i' % ii] = terp.UnivariateSpline(zdat[lowZLoc:upZLoc],
                                                       np.array(stigLib[ii]['sigmay'])[lowZLoc:upZLoc],
                                                       ext='zeros')(zVal)
        # set regions outside of usable interpolation area to very unreasonable sigma values
        sigCalX['chan%i' % ii][sigCalX['chan%i' % ii] == 0] = 1e5  # np.nan_to_num(np.inf)
        sigCalY['chan%i' % ii][sigCalY['chan%i' % ii] == 0] = 1e5  # np.nan_to_num(np.inf)

    for mi in range(numMols):
        chans = np.where(fres['planeCounts'][mi] > 0)[0]
        errX, errY = 0, 0
        for ci in chans:
            wx = 1./(fres['fitError_sigmaxPlane%i' % chanPlane[ci]][mi])**2
            wy = 1./(fres['fitError_sigmayPlane%i' % chanPlane[ci]][mi])**2
            errX += wx*(fres['fitResults_sigmaxPlane%i' % chanPlane[ci]][mi] - sigCalX['chan%i' % ci])**2
            errY += wy*(fres['fitResults_sigmayPlane%i' % chanPlane[ci]][mi] - sigCalY['chan%i' % ci])**2
        try:
            z[mi] = zVal[np.nanargmin(errX + errY)]
        except:
            print('No sigmas in correct plane for this molecule')

    return z

def coalesceDict(inD, assigned):  # , notKosher=None):
    """
    Agregates clumps to a single event
    Note that this will evaluate the lazy pipeline events and add them into the dict as an array, not a code
    object.
    Also note that copying a large dictionary can be rather slow, and a structured ndarray approach may be preferable.

    Args:
        inD: input dictionary containing fit results
        assigned: clump assignments to be coalesced

    Returns:
        fres: output dictionary containing the coalesced results

    """
    NClumps = int(np.max(assigned))  # len(np.unique(assigned))  #

    fres = {}

    clist = [[] for i in xrange(NClumps)]
    for i, c in enumerate(assigned):
        clist[int(c-1)].append(i)


    for rkey in inD.keys():
        skey = rkey.split('_')

        if skey[0] == 'fitResults':
            fres[rkey] = np.empty(NClumps)
            errKey = 'fitError_' + skey[1]
            fres[errKey] = np.empty(NClumps)
            for i in xrange(NClumps):
                ci = clist[i]
                fres[rkey][i], fres[errKey][i] = pyDeClump.weightedAverage_(inD[rkey][ci], inD[errKey][ci], None)
        elif rkey == 'tIndex':
            fres[rkey] = np.empty(NClumps)
            for i in xrange(NClumps):
                ci = clist[i]
                fres['tIndex'][i] = inD['tIndex'][ci].min()

        elif rkey == 'whichChan':
            fres[rkey] = np.empty(NClumps, dtype=np.int32)
            if 'planeCounts' in inD.keys():
                fres['planeCounts'] = np.zeros((NClumps, inD['planeCounts'].shape[1]))
                for i in xrange(NClumps):
                    ci = clist[i]  # clump indices
                    cl = inD[rkey][ci]  # channel list of clumps

                    fres[rkey][i] = np.array(np.bincount(cl).argmax(), dtype=np.int32)  # set channel to mode

                    # if np.logical_and(len(np.unique(cl)) > 1, np.any([entry in cl for entry in notKosher])):

                    fres['planeCounts'][i][:] = inD['planeCounts'][ci][:].sum(axis=0).astype(np.int32)
                    #cind, counts = np.unique(cl, return_counts=True)
                    #fres['planeCounts'][i][cind] += counts.astype(np.int32)

            else:
                for i in xrange(NClumps):
                    ci = clist[i]
                    cl = inD[rkey][ci]

                    fres[rkey][i] = np.array(np.bincount(cl).argmax(), dtype=np.int32)  # set channel to mode

        elif rkey == 'planeCounts' or skey[0] == 'fitError':
            pass

        else:  # settle for the unweighted mean
            fres[rkey] = np.empty(NClumps)
            for i in xrange(NClumps):
                ci = clist[i]
                fres[rkey][i] = inD[rkey][ci].mean()

    return fres

class multiviewMapper:
    """

    multiviewMapper provides methods for registering multiview channels as acquired in multicolor or biplane imaging.
    Image frames for multiview data should have channels concatenated horizontally, such that the x-dimension is the
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

    def OnFoldAndMapXY(self, event):
        """
        OnFoldAndMap uses shiftmaps stored in metadata (by default) or loaded through the GUI to register multiview
        channels to the first channel.
        Args:
            event: GUI event

        Returns: nothing
            x- and y-positions will be registered to the first channel in the mappingFilter with shiftmap corrections
            applied (see foldX)

        """
        pipeline = self.visFr.pipeline

        try:  # load shiftmaps from metadata, if present
            shiftWallet = pipeline.mdh['Shiftmap']
        except AttributeError:
            try:  # load through GUI dialog
                fdialog = wx.FileDialog(None, 'Load shift field', wildcard='Shift Field file (*.sf)|*.sf',
                                        style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
                succ = fdialog.ShowModal()
                if (succ == wx.ID_OK):
                    fpath = fdialog.GetPath()
                    # load json
                    fid = open(fpath, 'r')
                    shiftWallet = json.load(fid)
                    fid.close()
            except:
                raise IOError('Shiftmaps not found in metadata and could not be loaded from file')

        numChan = pipeline.mdh['Multiview.NumROIs']
        # fold x-positions into the first channel
        foldX(pipeline)

        plotFolded(pipeline['x'], pipeline['y'],
                            pipeline['whichChan'], 'Raw')

        # apply shiftmaps
        applyShiftmaps(pipeline, shiftWallet, numChan)

        # create new data source NO LONGER NECESSARY, mapping filter will reg x and y automatically
        #from PYME.LMVis.inpFilt import fitResultsSource
        #fres = pipeline.selectedDataSource.resultsSource.fitResults
        #regFres = np.copy(fres)
        #regFres['fitResults']['x0'], regFres['fitResults']['y0'] = x, y
        #pipeline.addDataSource('RegisteredXY', fitResultsSource(regFres))
        #pipeline.selectDataSource('RegisteredXY')

    def OnCalibrateShifts(self, event):
        """

        OnRegisterMultiview generates multiview shiftmaps on bead-data. Only beads which show up in all channels are
        used to generate the shiftmap.

        Args:
            event: GUI event

        Returns: nothing
            Writes shiftmapWallet into a json formatted .sf file through a GUI dialog
        """
        from PYME.Analysis.points import twoColour
        pipeline = self.visFr.pipeline

        try:
            numChan = pipeline.mdh['Multiview.NumROIs']
        except AttributeError:
            raise AttributeError('You are either not looking at multiview Data, or your metadata is incomplete')

        # fold x position of channels into the first
        foldX(pipeline)

        plotFolded(pipeline['x'], pipeline['y'], pipeline['whichChan'], 'Raw')
        # sort in frame order
        I = pipeline['tIndex'].argsort()
        xsort, ysort = pipeline['x'][I], pipeline['y'][I]
        chanSort = pipeline['whichChan'][I]

        clumpRad = 2e3*pipeline.mdh['voxelsize.x']  # clump folded data within 2 pixels

        clumpID, keep = pairMolecules(pipeline['tIndex'][I], xsort, ysort, chanSort, clumpRad*np.ones_like(xsort),
                                          appearIn=np.arange(numChan), nFrameSep=pipeline['tIndex'].max())


        # only look at the ones which showed up in all channels
        x = xsort[keep]
        y = ysort[keep]
        Chan = chanSort[keep]
        clumpID = clumpID[keep]

        # Generate raw shift vectors (map of displacements between channels) for each channel
        molList = np.unique(clumpID)
        numMoles = len(molList)

        dx = np.zeros((numChan - 1, numMoles))
        dy = np.zeros_like(dx)
        dxErr = np.zeros_like(dx)
        dyErr = np.zeros_like(dx)
        xClump, yClump, xStd, yStd, xShifted, yShifted = [], [], [], [], [], []
        shiftWallet = {}
        # dxWallet, dyWallet = {}, {}
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
                shiftWallet['Chan0%s.X' % ii], shiftWallet['Chan0%s.Y' % ii] = spx.__dict__, spy.__dict__
                # dxWallet['Chan0%s' % ii], dyWallet['Chan0%s' % ii] = dxx, dyy

                # shift the clumps for plotting
                xShifted.append(xClump[ii] + spx(xClump[ii], yClump[ii]))
                yShifted.append(yClump[ii] + spy(xClump[ii], yClump[ii]))
            else:
                xShifted.append(xClump[ii])
                yShifted.append(yClump[ii])


        shiftWallet['shiftModel'] = '.'.join([spx.__class__.__module__, spx.__class__.__name__])

        applyShiftmaps(pipeline, shiftWallet, numChan)

        plotFolded(pipeline['x'], pipeline['y'],
                            pipeline['whichChan'], 'All beads after Registration')

        cStack = []
        for ci in range(len(xShifted)):
            cStack.append(ci*np.ones(len(xShifted[ci])))
        cStack = np.hstack(cStack)

        plotFolded(np.hstack(xClump), np.hstack(yClump), cStack, 'Unregistered Clumps')

        plotFolded(np.hstack(xShifted), np.hstack(yShifted), cStack, 'Registered Clumps')

        # save shiftmaps
        defFile = os.path.splitext(os.path.split(self.visFr.GetTitle())[-1])[0] + 'MultiView.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            fid = open(fpath, 'wb')
            json.dump(shiftWallet, fid)
            fid.close()

    def OnMapZ(self, event):
        pipeline = self.visFr.pipeline

        # get channel and color info
        try:
            numChan = pipeline.mdh['Multiview.NumROIs']
            try:
                chanColor = pipeline.mdh['Multiview.ChannelColor']  # Bewersdorf Biplane example: [0, 1, 1, 0]
            except AttributeError:
                chanColor = [0 for c in range(numChan)]
            try:
                chanPlane = pipeline.mdh['Multiview.ChannelPlane']  # Bewersdorf Biplane example: [0, 0, 1, 1]
            except AttributeError:
                chanPlane = [0 for c in range(numChan)]
            numPlanes = len(np.unique(chanPlane))
        except AttributeError:  # default to non-multiview options
            print('Defaulting to single plane, single color channel settings')
            numChan = 1
            chanColor = [0]
            numPlanes = 1

        try:  # load astigmatism calibrations from metadata, if present
            stigLib = pipeline.mdh['astigLib']
        except AttributeError:
            try:  # load through GUI dialog
                fdialog = wx.FileDialog(None, 'Load Astigmatism Calibration', #wildcard='Shift Field file (*.sf)|*.sf',
                                        wildcard='AstigMAPism file (*.am)|*.am', style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
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

        # add separate sigmaxy columns for each plane
        for pind in range(numPlanes):
            pMask = [chanPlane[p] == pind for p in pipeline.mapping['whichChan']]

            pipeline.addColumn('fitResults_sigmaxPlane%i' % pind, pMask*pipeline.mapping['fitResults_sigmax'])
            pipeline.addColumn('fitResults_sigmayPlane%i' % pind, pMask*pipeline.mapping['fitResults_sigmay'])
            pipeline.addColumn('fitError_sigmaxPlane%i' % pind, pMask*pipeline.mapping['fitError_sigmax'])
            pipeline.addColumn('fitError_sigmayPlane%i' % pind, pMask*pipeline.mapping['fitError_sigmay'])
            # replace zeros in fiterror with infs so their weights are zero
            pipeline.mapping['fitError_sigmaxPlane%i' % pind][pipeline.mapping['fitError_sigmaxPlane%i' % pind] == 0] = np.inf
            pipeline.mapping['fitError_sigmayPlane%i' % pind][pipeline.mapping['fitError_sigmayPlane%i' % pind] == 0] = np.inf

        ni = len(pipeline.mapping['whichChan'])
        fres = {}
        for pkey in pipeline.mapping.keys():
            fres[pkey] = pipeline.mapping[pkey]

        fres['planeCounts'] = np.zeros((ni, numChan), dtype=np.int32)
        for j in range(ni):
                #cind, counts = np.unique(cl, return_counts=True)
                #fres['planeCounts'][i][:] = 0  # zero everything since the array will be empty, and we don't know numChan
                fres['planeCounts'][j][fres['whichChan'][j]] += 1

        for cind in np.unique(chanColor):
            # copy keys and sort in order of frames
            I = np.argsort(fres['tIndex'])
            for pkey in fres.keys():
                fres[pkey] = fres[pkey][I]

            # make sure NaNs (awarded when there is no sigma in a given plane of a clump) do not carry over from when
            # ignored channel localizations were clumped by themselves
            for pp in range(numPlanes):
                fres['fitResults_sigmaxPlane%i' % pp][np.isnan(fres['fitResults_sigmaxPlane%i' % pp])] = 0
                fres['fitResults_sigmayPlane%i' % pp][np.isnan(fres['fitResults_sigmayPlane%i' % pp])] = 0

            # trick pairMolecules function by tweaking the channel vector
            planeInColorChan = np.copy(fres['whichChan'])
            chanColor = np.array(chanColor)
            ignoreChans = np.where(chanColor != cind)[0]
            igMask = [mm in ignoreChans.tolist() for mm in planeInColorChan]
            planeInColorChan[np.where(igMask)] = -9  # must be negative to be ignored

            # assign molecules to clumps
            clumpID, paired = pairMolecules(fres['tIndex'], fres['fitResults_x0'], fres['fitResults_y0'],
                                            planeInColorChan, deltaX=fres['fitError_x0'],
                                            appearIn=np.where(chanColor == cind)[0], nFrameSep=1)

            # make sure clumpIDs are contiguous from [0, numClumps)
            assigned = -1*np.ones_like(clumpID)
            clumpVec = np.unique(clumpID)
            for ci in range(len(clumpVec)):
                cMask = clumpID == clumpVec[ci]
                assigned[cMask] = ci + 1  #FIXME: cluster assignments currently must start from 1, which is mean.

            # coalesce clumped localizations into single data point
            fres = coalesceDict(fres, assigned)  # , ignoreChans)

        print('Clumped %i localizations' % (ni - len(fres['whichChan'])))

        # look up z-positions
        z = astigMAPism(fres, stigLib, chanPlane)
        fres['astigZ'] = z

        # make sure there is no z, so that focus will be added during addDataSource
        del fres['z']
        pipeline.addDataSource('Zmapped', cachingResultsFilter(fres))
        pipeline.selectDataSource('Zmapped')



def Plug(visFr):
    """Plugs this module into the gui"""
    multiviewMapper(visFr)
