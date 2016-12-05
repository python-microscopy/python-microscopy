
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
from PYME.Analysis.points.DeClump import deClump

from PYME.Analysis.points.astigmatism import astigTools
import os
from PYME.IO.FileUtils import nameUtils
import json
# import importlib

#import scipy.interpolate as terp #terp doesn't really tell us what it means
from scipy.interpolate import UnivariateSpline #as we only use this function, interpolate it directly

#from PYME.LMVis.inpFilt import cachingResultsFilter  # mappingFilter  # fitResultsSource

import logging
logger = logging.getLogger(__name__)


def foldX(pipeline):
    """

    At this point the origin of x should be the corner of the concatenated frame

    Args:
        pipeline:

    Returns: nothing
        Adds folded x-coordinates to the pipeline
        Adds channel assignments to the pipeline

    """
    from PYME.Analysis.points import multiview
    multiview.foldX(pipeline.selectedDataSource, pipeline.mdh)


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

    c = iter(plt.cm.Dark2(np.linspace(0, 1, nChan)))
    for ii in range(nChan):
        mask = (ii == multiviewChannels)
        plt.scatter(X[mask], Y[mask], edgecolors=next(c), facecolors='none', label='Chan #%i' % ii)
    plt.title(title)
    plt.legend()
    return

def pairMolecules(tIndex, x, y, whichChan, deltaX=[None], appearIn=np.arange(4), nFrameSep=5, returnPaired=True):
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
        returnPaired: boolean flag to return a boolean array where True indicates that the molecule is a member of a
            clump whose members span the appearIn channels.

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
    assigned = pyDeClump.findClumps(tIndex.astype(np.int32), x, y, deltaX, nFrameSep)
    # print assigned.min()

    # only look at clumps with localizations from each channel
    clumps = np.unique(assigned)

    # Note that this will never be a keep clump if an ignore channel is present...
    keptClumps = [np.array_equal(np.unique(whichChan[assigned == clumps[ii]]), appearIn) for ii in range(len(clumps))]
    #keptClumps = [(len(np.unique(whichChan[assigned == clumps[ii]])) >= appearances) for ii in range(len(clumps))]

   # don't clump molecules from the wrong channel (done by parsing modified whichChan to this function)
    ignoreChan = whichChan < 0
    numClump = np.max(assigned)
    igVec = np.arange(numClump + 1, numClump + 1 + sum(ignoreChan))
    # give ignored channel localizations unique clump assignments
    assigned[ignoreChan] = igVec

    if returnPaired:
        keptMoles = []
        # np.array_equal(clumps, np.arange(1, np.max(assigned) + 1)) evaluates to True
        # TODO: speed up following loop - quite slow for large N
        for elem in assigned:
            keptMoles.append(elem in clumps[np.where(keptClumps)])
        keep = np.where(keptMoles)
        return assigned, keep
    else:
        return assigned

def applyShiftmaps(pipeline, shiftWallet):
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
    from PYME.Analysis.points import multiview
    multiview.applyShiftmaps(pipeline.selectedDataSource, shiftWallet)

def astigMAPism(fres, astig_calibrations, chanPlane, chanColor):
    """
    Generates a look-up table of sorts for z based on sigma x/y fit results and calibration information. If a molecule
    appears on multiple planes, sigma values from both planes will be used in the look up.
    Args:
        fres: dictionary-like object containing relevant fit results
        astig_calibrations: list of astigmatism calibration dictionaries corresponding to each multiview channel, which are
            used to recreate shiftmap objects
        chanPlane: list of which plane each channel corresponds to, e.g. [0, 0, 1, 1]

    Returns:
        z: an array of z-positions for each molecule in nm (assuming proper units were used in astigmatism calibration)
        zerr: an array containing discrepancies between sigma values and the PSF calibration curves. Note that this
            array is in units of nm, but error may not be propagated from sigma fitResults properly as is.
    """
    # fres = pipeline.selectedDataSource.resultsSource.fitResults
    numMols = len(fres['x']) # there is no guarantee that fitResults_x0 will be present - change to x
    whichChan = np.array(fres['multiviewChannel'], dtype=np.int32)

    # astig_calibrations['zRange'] contains the extrema of acceptable z-positions looking over all channels

    # find overall min and max z values
    zrange = np.nan*np.ones(2)
    for astig_cal in astig_calibrations: #more idiomatic way of looping through list - also avoids one list access / lookup
        zrange = [np.nanmin(astig_cal['zRange'][0], zrange[0]), np.nanmax(astig_cal['zRange'][1], zrange[1])]

    # generate z vector for interpolation
    zVal = np.arange(zrange[0], zrange[1])

    sigCalX = {}
    sigCalY = {}

    z = np.zeros(numMols)
    zerr = 1e4*np.ones(numMols)

    #TODO - Is this a robust choice?
    smoothFac = 5*len(astig_calibrations[0]['z'])

    # generate look up table of sorts
    #import matplotlib.pyplot as plt
    #plt.figure()
    for ii in range(len(chanPlane)):
        zdat = np.array(astig_calibrations[ii]['z'])
        # find indices of range we trust
        zrange = astig_calibrations[ii]['zRange']
        lowsubZ , upsubZ = np.absolute(zdat - zrange[0]), np.absolute(zdat - zrange[1])
        lowZLoc = np.argmin(lowsubZ)
        upZLoc = np.argmin(upsubZ)

        sigCalX['chan%i' % ii] = UnivariateSpline(zdat[lowZLoc:upZLoc],
                                                       np.array(astig_calibrations[ii]['sigmax'])[lowZLoc:upZLoc],
                                                       ext='const', s=smoothFac)(zVal)  #  ext='const', s=smoothFac)(zVal)
                                                            # bbox=astig_calibrations['PSF%i' % ii]['zrange'], ext='zeros')(zVal)
        sigCalY['chan%i' % ii] = UnivariateSpline(zdat[lowZLoc:upZLoc],
                                                       np.array(astig_calibrations[ii]['sigmay'])[lowZLoc:upZLoc],
                                                       ext='const', s=smoothFac)(zVal)  # ext='const', s=smoothFac)(zVal)
        # set regions outside of usable interpolation area to very unreasonable sigma values
        #sigCalX['chan%i' % ii][sigCalX['chan%i' % ii] == 0] = 1e5  # np.nan_to_num(np.inf)
        #sigCalY['chan%i' % ii][sigCalY['chan%i' % ii] == 0] = 1e5  # np.nan_to_num(np.inf)

        #plt.plot(zVal, sigCalX['chan%i' % ii])
        #plt.plot(zVal, sigCalY['chan%i' % ii])


    failures = 0
    for mi in range(numMols):
        chans = np.where(fres['probe'][mi] == chanColor)[0]
        errX, errY = 0, 0
        wSum = 0
        #plt.figure(10)
        #plt.subplot(1, 2, 2)
        #sigxList = []
        #sigyList = []
        for ci in chans:
            if not np.isnan(fres['sigmaxPlane%i' % chanPlane[ci]][mi]):
                #plt.plot(zVal, sigCalX['chan%i' % ci], label='$\sigma_x$, chan %i' % ci)
                #plt.plot(zVal, sigCalY['chan%i' % ci], label='$\sigma_y$, chan %i' % ci)
                #sigxList.append(fres['sigmaxPlane%i' % chanPlane[ci]][mi])
                #sigyList.append(fres['sigmayPlane%i' % chanPlane[ci]][mi])

                wX = 1./(fres['error_sigmaxPlane%i' % chanPlane[ci]][mi])**2
                wY = 1./(fres['error_sigmayPlane%i' % chanPlane[ci]][mi])**2
                wSum += (wX + wY)
                errX += wX*(fres['sigmaxPlane%i' % chanPlane[ci]][mi] - sigCalX['chan%i' % ci])**2
                errY += wY*(fres['sigmayPlane%i' % chanPlane[ci]][mi] - sigCalY['chan%i' % ci])**2
        try:
            err = (errX + errY)/wSum
            minLoc = np.nanargmin(err)
            z[mi] = -zVal[minLoc]
            zerr[mi] = np.sqrt(err[minLoc])

            #if len(sigxList)>1:
            #    plt.scatter(z[mi]*np.ones(len(sigxList)), sigxList, label='$\sigma_x$', c='red')
            #    plt.scatter(z[mi]*np.ones(len(sigxList)), sigyList, label='$\sigma_y$', c='black')
            #    plt.legend()
            #    plt.subplot(1, 2, 1)
            #    plt.plot(zVal, errX/wSum, label='error X')
            #    plt.plot(zVal, errY/wSum, label='error Y')
            #    plt.plot(zVal, err, label='Total Error')
            #    plt.xlabel('Z-position [nm]')
            #    plt.ylabel(r'Error [nm$^2$]')
            #    plt.legend()
            #    plt.show()
            #plt.clf()


        except (TypeError, ValueError, ZeroDivisionError):
            # TypeError if err is scalar 0, ValueError if err is all NaNs, ZeroDivErr if wSum and errX are both zero
            failures += 1

    #plt.hist(-z)
    #plt.xlabel('Z-position from astigmatism [nm]')
    #plt.ylabel('Counts [unitless] or Sigma [nm]')

    print('%i localizations did not have sigmas in acceptable range/planes (out of %i)' % (failures, numMols))

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.hist(z)
    #plt.show()
    return z, zerr





def coalesceDict(inD, assigned, keys, weightList):  # , notKosher=None):
    """
    Agregates clumps to a single event
    Note that this will evaluate the lazy pipeline events and add them into the dict as an array, not a code
    object.
    Also note that copying a large dictionary can be rather slow, and a structured ndarray approach may be preferable.
    DB - we should never have a 'large' dictionary (ie there will only ever be a handful of keys)

    Args:
        inD: input dictionary containing fit results
        assigned: clump assignments to be coalesced
        keys: list whose elements are strings corresponding to keys to be copied from the input to output dictionaries
        weightList: list whose elements describe the weightings to use when coalescing corresponding keys. A scalar
            weightList entry flags for an unweighted mean is performed during coalescence. Alternatively, if a
            weightList element is a vector it will be used as the weights, and a string will be used as a key to extract
            weights from the input dictionary.

    Returns:
        fres: output dictionary containing the coalesced results

    """
    NClumps = int(np.max(assigned))  # len(np.unique(assigned))  #

    clumped = {}

    clist = [[] for i in xrange(NClumps)]
    for i, c in enumerate(assigned):
        clist[int(c-1)].append(i)

    # loop through keys
    for ki in range(len(keys)):
        rkey = keys[ki]

        #we will aggregate into this and then add to dictionary.
        #performing dictionary lookups (as previously done) on each loop iteration is expensive
        var = np.empty(NClumps)
        clumped[rkey] = var

        # determine if weights need to be propogated into new dictionary
        keepWeights = isinstance(weightList[ki], basestring) # testing __class__ is not idomatic (and will fail in python if we end up with unicode or orther derived string types)
        if keepWeights:
            weights = inD[weightList[ki]]
            clumped[weightList[ki]] = np.empty(NClumps)
        else:
            weights = weightList[ki]


        if np.isscalar(weights):
            # if single value is given as weight, take an unweighted mean
            for i in xrange(NClumps):
                clumped[rkey][i] = np.mean(inD[rkey][clist[i]])
                #var[i] = np.mean(inD[rkey][clist[i]])
        elif weights is None:
            # if None has been passed as the weight for this key, take the minimum
            for i in xrange(NClumps):
                clumped[rkey][i] = np.min(inD[rkey][clist[i]])
                #var[i] = np.min(inD[rkey][clist[i]])
        else:
            # if weights is an array, take weighted average
            errVec = np.empty(NClumps)
            for i in xrange(NClumps):
                ci = clist[i]
                clumped[rkey][i], errVec[i] = pyDeClump.weightedAverage_(inD[rkey][ci], weights[ci], None)
                #var[i], errVec[i] = pyDeClump.weightedAverage_(inD[rkey][ci], weights[ci], None)

        #clumped[rkey] = var
        # propagate fitErrors into new dictionary
        if keepWeights:
            clumped[weightList[ki]] = errVec


    return clumped




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
        self.pipeline = visFr.pipeline
        self.clump_gap_tolerance = 1 # the number of frames that can be skipped for a clump to still be considered a single clump
        self.clump_radius_scale = 2.0 # the factor with which to multiply error_x by to determine a radius in which points belong to the same clump
        self.clump_radius_offset = 150. # an offset in nm to add to the the clump detection radius (useful for detection before shift correction)

        logging.debug('Adding menu items for multi-view manipulation')

        visFr.AddMenuItem('Multiview', 'Calibrate Shifts', self.OnCalibrateShifts,
                          helpText='Extract a shift field from bead measurements')

        visFr.AddMenuItem('Multiview', itemType='separator')

        visFr.AddMenuItem('Multiview', 'Fold Channels', self.OnFold)
        visFr.AddMenuItem('Multiview', 'Shift correct folded channels', self.OnCorrectFolded)

        visFr.AddMenuItem('Multiview', 'Group localizations', self.OnMergeClumps)

        visFr.AddMenuItem('Multiview', 'Map astigmatic Z', self.OnMapZ,
                          helpText='Look up z value for astigmatic 3D, using a multi-view aware correction')

        visFr.AddMenuItem('Multiview', itemType='separator')
        visFr.AddMenuItem('Multiview', 'Find clumps', self.OnFindClumps)

        visFr.AddMenuItem('Multiview', 'Map XY', self.OnFoldAndMapXY,
                          helpText='Fold channels and correct shifts')

        visFr.AddMenuItem('Multiview', 'Old clump and map function', self.OnGroupAndMapZ,
                          helpText='Look up z value for astigmatic 3D, using a multi-view aware correction')

    def OnFold(self, event=None):
        foldX(self.pipeline)

    def OnCorrectFolded(self, event=None):
        pipeline = self.pipeline

        if 'FIXMESiftmap' in pipeline.mdh.keys():  # load shiftmaps from metadata, if present
            shiftWallet = pipeline.mdh['FIXMEShiftmap'] #FIXME: break this for now
        else:
            fdialog = wx.FileDialog(None, 'Load shift field', wildcard='Shift Field file (*.sf)|*.sf',
                                    style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                fpath = fdialog.GetPath()
                # load json
                with open(fpath, 'r') as fid:
                    shiftWallet = json.load(fid)
            else:
                raise RuntimeError('Shiftmaps not found in metadata and could not be loaded from file')

        numChan = pipeline.mdh['Multiview.NumROIs']

        applyShiftmaps(pipeline, shiftWallet)


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
        if not 'multiviewChannel' in self.pipeline.keys():
            self.OnFold()

        plotFolded(self.pipeline['x'], self.pipeline['y'],
                  self.pipeline['multiviewChannel'], 'Raw')

        self.OnCorrectFolded()



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
        pipeline = self.pipeline

        try:
            numChan = pipeline.mdh['Multiview.NumROIs']
        except AttributeError:
            raise AttributeError('You are either not looking at multiview Data, or your metadata is incomplete')

        rad_dlg = wx.NumberEntryDialog(None, 'Search Radius In Pixels', 'rad [pix]', 'rad [pix]', 1, 0, 9e9)
        rad_dlg.ShowModal()
        clumpRad = rad_dlg.GetValue()*1e3*pipeline.mdh['voxelsize.x']  # clump folded data within X pixels
        # fold x position of channels into the first
        foldX(pipeline)

        plotFolded(pipeline['x'], pipeline['y'], pipeline['multiviewChannel'], 'Raw')
        # sort in frame order
        I = pipeline['tIndex'].argsort()
        xsort, ysort = pipeline['x'][I], pipeline['y'][I]
        chanSort = pipeline['multiviewChannel'][I]

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

        applyShiftmaps(pipeline, shiftWallet)

        plotFolded(pipeline['x'], pipeline['y'],
                            pipeline['multiviewChannel'], 'All beads after Registration')

        cStack = []
        for ci in range(len(xShifted)):
            cStack.append(ci*np.ones(len(xShifted[ci])))
        cStack = np.hstack(cStack)

        # calculate standard deviation within clump before and after shift
        unRegStd = np.empty((numMoles, 2))
        regStd = np.empty_like(unRegStd)
        for mi in range(numMoles):
            # TODO: pull inline loops out, this is just being lazy
            unRegStd[mi, :] = np.std([xClump[cc][mi] for cc in range(numChan)]), np.std([yClump[cc][mi] for cc in range(numChan)])
            regStd[mi, :] = np.std([xShifted[cc][mi] for cc in range(numChan)]), np.std([yShifted[cc][mi] for cc in range(numChan)])

        print('Avg std(X) within clumps: unreg= %f, reg =  %f' % (unRegStd[:, 0].mean(), regStd[:, 0].mean()))
        print('Avg std(Y) within clumps: unreg= %f, reg = %f' % (unRegStd[:, 1].mean(), regStd[:, 1].mean()))

        plotFolded(np.hstack(xClump), np.hstack(yClump), cStack, 'Unregistered Clumps')

        plotFolded(np.hstack(xShifted), np.hstack(yShifted), cStack, 'Registered Clumps')

        # save shiftmaps
        #FIXME - Getting the filename through the title is super fragile - should not use pipeline.filename (or similar) instead
        defFile = os.path.splitext(os.path.split(self.pipeline.filename)[-1])[0] + 'MultiView.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
            wildcard='Shift Field file (*.sf)|*.sf', style=wx.SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            fid = open(fpath, 'wb')
            json.dump(shiftWallet, fid)
            fid.close()

    def OnFindClumps(self, event=None):
        from PYME.Analysis.points import multiview
        try:
            multiview.multicolorFindClumps(self.pipeline.selectedDataSource, self.clump_gap_tolerance,
                             self.clump_radius_scale, self.clump_radius_offset)
        except RuntimeError:
            print('running single-color clump finding')
            multiview.findClumps(self.pipeline.selectedDataSource, self.clump_gap_tolerance,
                             self.clump_radius_scale, self.clump_radius_offset)


    def OnMergeClumps(self, event=None):
        from PYME.Analysis.points import multiview

        if not 'clumpIndex' in self.pipeline.keys():
            logger.debug('No clumps found - running FindClumps')
            self.OnFindClumps()

        numChan = self.pipeline.mdh.getOrDefault('Multiview.NumROIs', 1)

        grouped = multiview.mergeClumps(self.pipeline.selectedDataSource, numChan)

        self.pipeline.addDataSource('Grouped', grouped)
        self.pipeline.selectDataSource('Grouped')


    def OnMapZ(self, event=None, useMD = True):
        from PYME.IO import unifiedIO
        pipeline = self.pipeline

        # FIXME - Rename metadata key to be more reasonable
        stigLoc = pipeline.mdh.getOrDefault('Analysis.AstigmatismMapID', None)

        if (not stigLoc is None) and useMD:
            s = unifiedIO.read(stigLoc)
            astig_calibrations = json.loads(s)
        else:
            fdialog = wx.FileDialog(None, 'Load Astigmatism Calibration', wildcard='Astigmatism map (*.am)|*.am',
                                    style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                fpath = fdialog.GetPath()

                fdialog.Destroy()
                # load json
                with open(fpath, 'r') as fid:
                    astig_calibrations = json.load(fid)
            else:
                fdialog.Destroy()
                logger.info('User canceled astigmatic calibration selection')
                return

        z, zerr = astigTools.lookup_astig_z(pipeline, astig_calibrations, plot=False)

        pipeline.addColumn('astigZ', z)
        pipeline.addColumn('zLookupError', zerr)

        pipeline.selectedDataSource.setMapping('z', 'focus*foreShort + astigZ')

        pipeline._process_colour()

        self.visFr.RefreshView()
        self.visFr.CreateFoldPanel()

    def OnGroupAndMapZ(self, event):
        pipeline = self.pipeline

        # get channel and color info
        # DB - Fixed to use idomatic way of getting parameters with default values - catching an attribute error is fragile
        # as some metadata handlers throw KeyError (and all probably should throw KeyErrors)
        numChan = pipeline.mdh.getOrDefault('Multiview.NumROIs', 1)
        chanColor = np.array(pipeline.mdh.getOrDefault('Multiview.ChannelColor', np.zeros(numChan, 'i'))).astype('i')

        #FIXME - 'plane' assignment does not make any sense - this should not be necessary
        chanPlane = pipeline.mdh.getOrDefault('Multiview.ChannelPlane', np.zeros(numChan, 'i'))
        numPlanes = len(np.unique(chanPlane))


        try:  # load astigmatism calibrations, if listed in metadata
            # FIXME: this is only set up to pull a local copy at the moment
            # FIXME - Rename metadata key to be more reasonable
            stigLoc = pipeline.mdh['AstigmapID']
            fid = open(stigLoc, 'r') #FIXME - this file desciptor is never closed!
            astig_calibrations = json.load(fid)
        except (AttributeError, IOError):
            try:  # load through GUI dialog
                fdialog = wx.FileDialog(None, 'Load Astigmatism Calibration', wildcard='Astigmatism map (*.am)|*.am',
                                        style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
                succ = fdialog.ShowModal()
                if (succ == wx.ID_OK):
                    fpath = fdialog.GetPath()
                    # load json
                    fid = open(fpath, 'r')
                    astig_calibrations = json.load(fid)
                    fid.close()
            except:
                #FIXME - blanket except clauses are never a good idea, and even if we were going to use one
                # it's unclear that an IO error is appropriate
                raise IOError('Astigmatism sigma-Z mapping information not found')

        # make sure we have already made channel assignments:
        if 'multiviewChannel' not in pipeline.mapping.keys():
            logger.warn('folding multi-view data without applying shiftmaps')
            foldX(pipeline)


        # add separate sigmaxy columns for each plane
        for pind in range(numPlanes):
            pMask = np.array([chanPlane[p] == pind for p in pipeline.mapping['multiviewChannel']])

            pipeline.addColumn('sigmaxPlane%i' % pind, pMask*pipeline.mapping['fitResults_sigmax'])
            pipeline.addColumn('sigmayPlane%i' % pind, pMask*pipeline.mapping['fitResults_sigmay'])

            # work around invalid modification of pipeline variables, and change from setting infs to setting a large
            # -ve sigma as a flag for missing data.
            sigx = -1e4*(1-pMask) + pMask*pipeline.mapping['fitError_sigmax']
            pipeline.addColumn('error_sigmaxPlane%i' % pind, sigx)
            sigy = -1e4*(1 - pMask) + pMask * pipeline.mapping['fitError_sigmay']
            pipeline.addColumn('error_sigmayPlane%i' % pind, sigy)

            # replace zeros in fiterror with infs so their weights are zero
            # FIXME - It is not a good idea (performance wise) to use NaNs
            # as it will cause a floating point error on each pass of the loop - could be orders of magnitude slower than
            # just using zero (or very small) weights.
            # FIXME - You absolutely CANNOT assign to pipeline elements like this! Once added as a column, mappings are
            # read-only
            #pipeline.mapping['error_sigmaxPlane%i' % pind][pipeline.mapping['error_sigmaxPlane%i' % pind] == 0] = np.inf
            #pipeline.mapping['error_sigmayPlane%i' % pind][pipeline.mapping['error_sigmayPlane%i' % pind] == 0] = np.inf

        #ni = len(pipeline.mapping['multiviewChannel'])
        #wChan = np.copy(pipeline.mapping['multiviewChannel']) #Is the copy necessary?
        ni = len(pipeline['x']) #look at x rather than multiviewChannel to reduce evaluation cost (x is also always guaranteed to be there) TODO - add a length function to pipeline

        ## Moved to folding
        #probe = [chanColor[wChan[mi]] for mi in range(ni)]
        #probe = chanColor[wChan] #should be better performance
        #pipeline.addColumn('probe', probe)

        #TODO - why are we copying the pipeline output?
        fres = {}
        for pkey in pipeline.mapping.keys():
            fres[pkey] = pipeline.mapping[pkey]

        keys_to_aggregate = ['x', 'y', 't', 'probe', 'tIndex', 'multiviewChannel']
        keys_to_aggregate += ['sigmax_%d' % chan for chan in range(numPlanes)]
        keys_to_aggregate += ['sigmay_%d' % chan for chan in range(numPlanes)]

        # pair fit results and errors for weighting
        aggregation_weights = ['error_' + k if 'error_' + k in fres.keys() else None for k in keys_to_aggregate]

        #Moved following outside loop as it is common to both colour channels

        # copy keys and sort in order of frames
        I = np.argsort(fres['tIndex'])
        for pkey in fres.keys():
            fres[pkey] = fres[pkey][I]

        # make sure NaNs (awarded when there is no sigma in a given plane of a clump) do not carry over from when
        # ignored channel localizations were clumped by themselves
        # TODO - is this the correct thing to do?
        for pp in range(numPlanes):
            fres['sigmaxPlane%i' % pp][np.isnan(fres['sigmaxPlane%i' % pp])] = 0
            fres['sigmayPlane%i' % pp][np.isnan(fres['sigmayPlane%i' % pp])] = 0

        #TODO - Do we really want / need to do this separately for the colour channels (this eliminates the possibility of doing ratiometric detection)
        for cind in np.unique(chanColor):
            # trick pairMolecules function by tweaking the channel vector
            planeInColorChan = np.copy(fres['multiviewChannel'])

            igMask = fres['probe'] != cind
            planeInColorChan[igMask] = -9  # must be negative to be ignored

            # assign molecules to clumps
            #TODO - a 2-pixel radius will break badly when emitter densities increase - should we base this on 'error_x' instead?
            # Yes, basing this on error_x makes sense, potentially with the inclusion of a offset based on shiftmap precision, i.e.
            # if the stddev within clumps after applying shiftmap is 15nm, add this in quadrature to 'error_x'. Though this would only be
            # helpful if you have a bad shiftmap.
            clumpRad = 1e3*pipeline.mdh['voxelsize.x']*np.ones_like(fres['x'])  # clump folded data within 2 pixels #fres['fitError_x0'],

            clumpID = pairMolecules(fres['tIndex'], fres['x'], fres['y'],
                                            planeInColorChan, deltaX=clumpRad,  # fres['fitError_x0'],
                                            appearIn=np.where(chanColor == cind)[0], nFrameSep=1, returnPaired=False)

            # TODO - Why do clumpIDs need to be contiguous (i.e. can we eliminate this step)
            # FIXME - if we want to eliminate this step, need to change NClumps = int(np.max(assigned)) in coalesceDict
            # make sure clumpIDs are contiguous from [0, numClumps)
            assigned = -1*np.ones_like(clumpID)
            clumpVec = np.unique(clumpID)
            for ci in range(len(clumpVec)):
                cMask = clumpID == clumpVec[ci]
                assigned[cMask] = ci + 1

            # coalesce clumped localizations into single data point
            fres = coalesceDict(fres, assigned, keys_to_aggregate, aggregation_weights)

        print('Clumped %i localizations' % (ni - len(fres['multiviewChannel'])))

        # look up z-positions
        z, zerr = astigMAPism(fres, astig_calibrations, chanPlane, chanColor)
        fres['astigZ'] = z
        fres['zLookupError'] = zerr

        # make sure there is no z, so that focus will be added during addDataSource
        if 'z' in fres.keys():
            del fres['z']
        pipeline.addDataSource('Zmapped', cachingResultsFilter(fres))
        pipeline.selectDataSource('Zmapped')
        pipeline._process_colour()



def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.multiview = multiviewMapper(visFr)
