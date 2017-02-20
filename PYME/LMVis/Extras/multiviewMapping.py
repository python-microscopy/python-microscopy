
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


import logging
logger = logging.getLogger(__name__)


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


class multiviewMapper:
    """

    multiviewMapper provides methods for registering multiview channels as acquired in multicolor or biplane imaging.
    Image frames for multiview data should have channels concatenated horizontally, such that the x-dimension is the
    only dimension that needs to be folded into the first channel.
    The shiftmaps are functions that interface like bivariate splines, but can be recreated from a dictionary of their
    fit-model parameters, which are stored in a dictionary in the shiftmap object.
    In the multiviewMapper class, shiftmaps for multiview data sources are stored in a dictionary of dictionaries.
    Each shiftmap is stored as a dictionary so that it can be easily written into a human-readable json file.

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

        visFr.AddMenuItem('Multiview', 'Find points from same molecule', self.OnFindClumps)
        visFr.AddMenuItem('Multiview', 'Group found points', self.OnMergeClumps)

        visFr.AddMenuItem('Multiview', 'Map astigmatic Z', self.OnMapZ,
                          helpText='Look up z value for astigmatic 3D, using a multi-view aware correction')

        visFr.AddMenuItem('Multiview', 'Check astigmatic PSF Calibration', self.OnCheckAstigCalibration)

    def OnFold(self, event=None):
        """
        See multiview.foldX. At this point the origin of x should be the corner of the concatenated frame

        Parameters
        ----------

            None, but requires metadata.

        Notes
        -----

        """

        from PYME.recipes.localisations import MultiviewFold

        recipe = self.pipeline.recipe
        recipe.add_module(MultiviewFold(recipe, inputName=self.pipeline.selectedDataSourceKey,
                                                      outputName='folded'))
        recipe.execute()
        self.pipeline.selectDataSource('folded')

    def OnCorrectFolded(self, event=None):
        """
        Applies chromatic shift correction to folded localization data that was acquired with an
        image splitting device, but localized without splitter awareness.
        See recipes.localizations.MultiviewShiftCorrect.

        Parameters
        ----------

            None

        Notes
        -----

        """

        from PYME.recipes.localisations import MultiviewShiftCorrect
        pipeline = self.pipeline
        recipe = self.pipeline.recipe

        if 'FIXMESiftmap' in pipeline.mdh.keys():  # load shiftmaps from metadata, if present
            fpath = pipeline.mdh['FIXMEShiftmap'] #FIXME: break this for now
        else:
            fdialog = wx.FileDialog(None, 'Load shift field', wildcard='Shift Field file (*.sf)|*.sf',
                                    style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
            succ = fdialog.ShowModal()

            if (succ == wx.ID_OK):
                fpath = fdialog.GetPath()
            else:
                raise RuntimeError('Shiftmaps not found in metadata and could not be loaded from file')

        recipe.add_module(MultiviewShiftCorrect(recipe, inputName=pipeline.selectedDataSourceKey,
                          shiftMapLocation=fpath, outputName='shift_corrected'))
        recipe.execute()
        self.pipeline.selectDataSource('shift_corrected')

    def OnCalibrateShifts(self, event):
        """

        Generates multiview shiftmaps on bead-data. Only beads which show up in all channels are
        used to generate the shiftmap.

        Parameters
        ----------

            searchRadius: Radius within which to clump bead localizations that appear in all channels. Units of pixels.

        Notes
        -----

        """
        from PYME.Analysis.points import twoColour
        from PYME.Analysis.points import multiview
        pipeline = self.pipeline

        try:
            numChan = pipeline.mdh['Multiview.NumROIs']
        except AttributeError:
            raise AttributeError('You are either not looking at multiview Data, or your metadata is incomplete')

        rad_dlg = wx.NumberEntryDialog(None, 'Search Radius In Pixels', 'rad [pix]', 'rad [pix]', 1, 0, 9e9)
        rad_dlg.ShowModal()
        clumpRad = rad_dlg.GetValue()*1e3*pipeline.mdh['voxelsize.x']  # clump folded data within X pixels
        # fold x position of channels into the first
        multiview.foldX(pipeline.selectedDataSource, pipeline.mdh, inject=True, chroma_mappings=True)

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

        multiview.applyShiftmaps(pipeline.selectedDataSource, shiftWallet)

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
        """

        Determines which localizations are likely to be the same molecule and assigns them the same label.

        Parameters
        ----------

            gap_tolerance: number of frames acceptable for a molecule to go MIA and still be called the same molecule
                when it returns
            radius_scale: multiplicative factor applied to the error_x term in deciding search radius for pairing
            radius_offset: term added to radius_scale*error_x to set search radius
            probeAwareClumping: boolean flag to tell FindClumps recipe module to cluster each probe separately


        Notes
        -----

        """
        from PYME.recipes.localisations import FindClumps
        recipe = self.pipeline.recipe
        recipe.add_module(FindClumps(recipe, inputName=self.pipeline.selectedDataSourceKey, outputName='with_clumps',
                                     gapTolerance=self.clump_gap_tolerance, radiusScale=self.clump_radius_scale,
                                     radius_offset_nm=self.clump_radius_offset, probeAwareClumping=True))
        recipe.execute()
        self.pipeline.selectDataSource('with_clumps')

    def OnMergeClumps(self, event=None):
        """

        Coalesces clusters of localization data considered to be the same molecule. See
        recipes.localizations.MergeClumps.

        Parameters
        ----------

            None


        Notes
        -----

        """
        from PYME.recipes.localisations import FindClumps, MergeClumps

        if not 'clumpIndex' in self.pipeline.keys():
            logger.debug('No clumps found - running FindClumps')
            self.OnFindClumps()

        recipe = self.pipeline.recipe

        recipe.add_module(MergeClumps(recipe, inputName='with_clumps', outputName='clumped'))

        recipe.execute()
        self.pipeline.selectDataSource('clumped')

    def OnMapZ(self, event=None, useMD = True):
        """

        Uses sigmax and sigmay values from astigmatic fits to look up a z-position using calibration curves.
        See Analysis.points.astigmatism.astigTools.lookup_astig_z

        Parameters
        ----------

            None


        Notes
        -----

        """

        from PYME.recipes.localisations import MapAstigZ
        #from PYME.IO import unifiedIO
        pipeline = self.pipeline

        # FIXME - Rename metadata key to be more reasonable
        stigLoc = pipeline.mdh.getOrDefault('Analysis.AstigmatismMapID', None)

        if (not stigLoc is None) and useMD:
            #s = unifiedIO.read(stigLoc)
            #astig_calibrations = json.loads(s)
            pathToMap = stigLoc
        else:
            fdialog = wx.FileDialog(None, 'Load Astigmatism Calibration', wildcard='Astigmatism map (*.am)|*.am',
                                    style=wx.OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                fpath = fdialog.GetPath()

                fdialog.Destroy()
                # load json
                #with open(fpath, 'r') as fid:
                #    astig_calibrations = json.load(fid)
                pathToMap = fpath
            else:
                fdialog.Destroy()
                logger.info('User canceled astigmatic calibration selection')
                return

        recipe = self.pipeline.recipe
        recipe.add_module(MapAstigZ(recipe, inputName=self.pipeline.selectedDataSourceKey,
                                    astigmatismMapLocation=pathToMap, outputName='z_mapped'))
        recipe.execute()
        self.pipeline.selectDataSource('z_mapped')

        pipeline._process_colour()

        self.visFr.RefreshView()
        self.visFr.CreateFoldPanel()

    def OnCheckAstigCalibration(self, event=None):
        """
        For use with dyes on a coverslip.
        NB localizations from transient frames should be filtered prior to calling this function

        Parameters
        ----------
        event: GUI event

        Returns
        -------

        """
        import matplotlib.pyplot as plt

        pipeline = self.pipeline
        # sort localizations according to frame
        I = np.argsort(pipeline['t'])
        zsort = pipeline['astigZ'][I]
        try:
            focus = pipeline['focus'][I]
        except AttributeError:
            raise UserWarning('CheckAstigCalibration requires ProtocolFocus events OR StackSettings metadata (StepSize, FramesPerStep)')

        # find unique of ints to avoid floating point issues
        uni, counts = np.unique(focus.astype(int), return_counts=True)

        cycles = len(uni)
        mu = np.empty(cycles)
        stdDev = np.empty(cycles)

        colors = iter(plt.cm.Dark2(np.remainder(np.linspace(0, 2*np.pi, cycles + 1), np.pi*np.ones(cycles + 1))))
        plt.figure()
        plt.xlabel('Z position [nm]')
        plt.ylabel('Counts')

        indi = 0
        for ci in range(len(uni)):
            indf = indi + counts[ci]
            slicez = zsort[indi:indf]
            mu[ci] = np.mean(slicez)
            stdDev[ci] = np.std(slicez)

            nc = next(colors)
            plt.hist(slicez, bins=range(int(min(slicez)), int(max(slicez) + 0.5*stdDev[ci]), int(0.5*stdDev[ci])), color=nc)

            indi = indf

        plt.figure()
        diffs = mu[:-1] - mu[1:]
        plt.bar(uni[:-1], diffs, width=(0.75*(uni[1] - uni[0])))
        plt.ylabel('Separation between steps')

        plt.figure()
        try:
            plt.plot(pipeline.mdh['StackSettings.StepSize']*np.array([1, 1]), [0, 1], color='black', label='Target Separation')
        except AttributeError:
            pass
        plt.scatter(diffs, 0.5*np.ones_like(diffs), label='Separations')
        plt.title('Std deviation of step separations = %.1f' % np.std(diffs))
        plt.legend(scatterpoints=1, loc=2)
        plt.show()


def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.multiview = multiviewMapper(visFr)
