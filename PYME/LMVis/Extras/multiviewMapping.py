
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
    xOG = pipeline['x']
    roiSizeNM = (pipeline.mdh['Multiview.ROISize'][1]*pipeline.mdh['voxelsize.x']*1000)  # voxelsize is in um
    xfold = xOG % roiSizeNM
    mvQuad = np.floor(xOG / roiSizeNM)

    pipeline.mapping.setMapping('xFolded', xfold)
    pipeline.mapping.setMapping('whichChan', mvQuad)
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

def pairMolecules(tIndex, x, y, whichChan, numChan, deltaX=None):
    """
    pairMolecules uses pyDeClump functions to group localization clumps into molecules for registration.

    Args:
        tIndex: from fitResults
        x: x positions of localizations AFTER having been folded into the first channel
        y: y positions of localizations
        whichChan: a vector containing channel assignments for each localization
        numChan: number of multiview channels
        deltaX: distance within which neighbors will be clumped is set by 2*deltaX[i])**2

    Returns:
        x and y positions of molecules that were clumped, which channel those localizations are from,
        and which clump they were assigned to.

    """
    # sort everything in frame order
    I = tIndex.argsort()
    tIndex = tIndex[I]
    x = x[I]
    y = y[I]
    whichChan = whichChan[I]

    # group within a certain distance, potentially based on localization uncertainty
    if not deltaX:
        dX = 100.*np.ones_like(x)
    # group localizations
    assigned = pyDeClump.findClumps(tIndex, x, y, dX)

    # only look at clumps with localizations from each channel
    clumps = np.unique(assigned)
    keptClumps = [np.array_equal(np.unique(whichChan[assigned == clumps[ii]]), np.arange(numChan)) for ii in range(len(clumps))]

    keptMoles = []
    # np.array_equal(clumps, np.arange(1, np.max(assigned) + 1)) evaluates to True
    for elem in assigned:
        keptMoles.append(elem in clumps[np.where(keptClumps)])
    keep = np.where(keptMoles)

    return x[keep], y[keep], whichChan[keep], assigned[keep]

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

        ID_REGISTER_MULTIVIEW = wx.NewId()
        visFr.extras_menu.Append(ID_REGISTER_MULTIVIEW, "Multiview - Register Channels")
        visFr.Bind(wx.EVT_MENU, self.OnRegisterMultiview, id=ID_REGISTER_MULTIVIEW)

        ID_MAP_MULTIVIEW = wx.NewId()
        visFr.extras_menu.Append(ID_MAP_MULTIVIEW, "Multiview - Map Z")
        visFr.Bind(wx.EVT_MENU, self.OnFoldAndMap, id=ID_MAP_MULTIVIEW)
        return

    def applyShiftmaps(self, x, y, numChan):
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

        xReg, yReg = [x[0]], [y[0]]
        for ii in range(1, numChan):
            xReg.append(x[ii] + shiftModel(dict=self.shiftWallet['Chan0%s.X' % ii]).ev(x[ii], y[ii]))
            yReg.append(y[ii] + shiftModel(dict=self.shiftWallet['Chan0%s.Y' % ii]).ev(x[ii], y[ii]))

        pipeline.mapping.setMapping('xReg', xReg)
        pipeline.mapping.setMapping('yReg', yReg)
        return

    def OnFoldAndMap(self, event):
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

        plotFolded(pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                            pipeline.mapping.__dict__['whichChan'], 'Raw')

        # organize x- and y-positions into list of arrays corresponding to channels
        xfold, yfold = [], []
        for ii in range(numChan):
            xfold.append(pipeline.mapping.__dict__['xFolded'][np.where(pipeline.mapping.__dict__['whichChan'] == ii)])
            yfold.append(pipeline['y'][np.where(pipeline.mapping.__dict__['whichChan'] == ii)])

        # apply shiftmaps
        self.applyShiftmaps(xfold, yfold, numChan)

        plotRegistered(pipeline.mapping.__dict__['xReg'], pipeline.mapping.__dict__['yReg'],
                            numChan, 'After Registration')

    def OnRegisterMultiview(self, event):
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

        # Match up molecules
        x, y, Chan, clumpID, = pairMolecules(pipeline['tIndex'], pipeline.mapping.__dict__['xFolded'], pipeline['y'],
                      pipeline.mapping.__dict__['whichChan'], numChan)  #, pipeline['error_x'])

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
        xfold, yfold = [], []
        for ii in range(numChan):
            xfold.append(pipeline.mapping.__dict__['xFolded'][np.where(pipeline.mapping.__dict__['whichChan'] == ii)])
            yfold.append(pipeline['y'][np.where(pipeline.mapping.__dict__['whichChan'] == ii)])

        plotRegistered(xfold, yfold, numChan, 'Raw')

        plotRegistered(xClump, yClump, numChan, 'Clumped')

        plotRegistered(pipeline.mapping.__dict__['xReg'], pipeline.mapping.__dict__['yReg'],
                            numChan, 'After Registration')




def Plug(visFr):
    """Plugs this module into the gui"""
    multiviewMapper(visFr)