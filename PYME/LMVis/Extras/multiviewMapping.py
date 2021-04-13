
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




class MultiviewMapper:
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

        # the following parameters are defaults which get (potentially) overridden by dialogs
        self.clump_gap_tolerance = 1 # the number of frames that can be skipped for a clump to still be considered a single clump
        self.clump_radius_scale = 2.0 # the factor with which to multiply error_x by to determine a radius in which points belong to the same clump
        self.clump_radius_offset = 150. # an offset in nm to add to the the clump detection radius (useful for detection before shift correction)

        logging.debug('Adding menu items for multi-view manipulation')

        visFr.AddMenuItem('Corrections>Multiview', 'Calibrate Shifts', self.OnCalibrateShifts,
                          helpText='Extract a shift field from bead measurements')

        visFr.AddMenuItem('Corrections>Multiview', itemType='separator')

        visFr.AddMenuItem('Corrections>Multiview', 'Fold Channels', self.OnFold)
        visFr.AddMenuItem('Corrections>Multiview', 'Shift correct folded channels', self.OnShiftCorrectFolded)

        visFr.AddMenuItem('Corrections>Multiview', 'Find points from same molecule', self.OnFindClumps)
        visFr.AddMenuItem('Corrections>Multiview', 'Group found points', self.OnMergeClumps)

        visFr.AddMenuItem('Corrections>Multiview', 'Map astigmatic Z', self.OnMapAstigmaticZ,
                          helpText='Look up z value for astigmatic 3D, using a multi-view aware correction')

        visFr.AddMenuItem('Corrections>Multiview', 'Check astigmatic PSF Calibration', self.OnCheckAstigmatismCalibration)

    def OnFold(self, event=None):
        """
        See multiview.foldX. At this point the origin of x should be the corner of the concatenated frame. Note that
        a probe key will be mapped into the data source to designate colour channel, but the colour channel will not be
        selectable in the GUI until after mergeClumps is called (due to the possibility of using this module with
        ratio-metric data).

        Parameters
        ----------

            None, but requires metadata.

        Notes
        -----

        """

        from PYME.recipes.multiview import Fold
        from PYME.recipes.tablefilters import FilterTable

        recipe = self.pipeline.recipe
        
        recipe.add_modules_and_execute([Fold(recipe,
                                             input_name=self.pipeline.selectedDataSourceKey, 
                                             output_name='folded')])
        
        self.pipeline.selectDataSource('folded')

    def OnShiftCorrectFolded(self, event=None):
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

        from PYME.recipes.multiview import ShiftCorrect
        pipeline = self.pipeline
        recipe = self.pipeline.recipe

        if 'FIXMESiftmap' in pipeline.mdh.keys():  # load shiftmaps from metadata, if present
            fpath = pipeline.mdh['FIXMEShiftmap'] #FIXME: break this for now
        else:
            fdialog = wx.FileDialog(None, 'Load shift field', wildcard='Shift Field file (*.sf)|*.sf',
                                    style=wx.FD_OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
            succ = fdialog.ShowModal()

            if (succ == wx.ID_OK):
                fpath = fdialog.GetPath()
            else:
                raise RuntimeError('Shiftmaps not found in metadata and could not be loaded from file')

        recipe.add_modules_and_execute([ShiftCorrect(recipe, input_name=pipeline.selectedDataSourceKey,
                          shift_map_path=fpath, output_name='shift_corrected'),])
        
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
        from PYME.recipes.multiview import CalibrateShifts

        # recipe = self.pipeline.recipe
        # # hold off auto-running the recipe until we configure things
        # recipe.trait_set(execute_on_invalidation=False)
        # try:
        #     calibration_module = CalibrateShifts(recipe, input_name=self.pipeline.selectedDataSourceKey,
        #                                      output_name='shiftmap')
        #
        #     recipe.add_module(calibration_module)
        #     if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
        #         return
        #
        #     recipe.execute()
        #     sm = recipe.namespace['shiftmap']
        # finally:  # make sure that we configure the pipeline recipe as it was
        #     recipe.trait_set(execute_on_invalidation=True)


        calibration_module = CalibrateShifts()
        if not calibration_module.configure_traits(kind='modal'):
            return
        
        sm = calibration_module.apply_simple(self.pipeline.selectedDataSource)


        # save the file
        defFile = os.path.splitext(os.path.split(self.pipeline.filename)[-1])[0] + 'MultiView.sf'

        fdialog = wx.FileDialog(None, 'Save shift field as ...',
                                wildcard='Shift Field file (*.sf)|*.sf', style=wx.FD_SAVE,
                                defaultDir=nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()

            sm.to_hdf(fpath, tablename='shift_map', metadata=sm.mdh)

        # multiview.applyShiftmaps(pipeline.selectedDataSource, shiftWallet)
        #
        # plotFolded(pipeline['x'], pipeline['y'],
        #                     pipeline['multiviewChannel'], 'All beads after Registration')
        #
        # cStack = []
        # for ci in range(len(xShifted)):
        #     cStack.append(ci*np.ones(len(xShifted[ci])))
        # cStack = np.hstack(cStack)
        #
        # # calculate standard deviation within clump before and after shift
        # unRegStd = np.empty((numMoles, 2))
        # regStd = np.empty_like(unRegStd)
        # for mi in range(numMoles):
        #     # TODO: pull inline loops out, this is just being lazy
        #     unRegStd[mi, :] = np.std([xClump[cc][mi] for cc in range(numChan)]), np.std([yClump[cc][mi] for cc in range(numChan)])
        #     regStd[mi, :] = np.std([xShifted[cc][mi] for cc in range(numChan)]), np.std([yShifted[cc][mi] for cc in range(numChan)])
        #
        # print('Avg std(X) within clumps: unreg= %f, reg =  %f' % (unRegStd[:, 0].mean(), regStd[:, 0].mean()))
        # print('Avg std(Y) within clumps: unreg= %f, reg = %f' % (unRegStd[:, 1].mean(), regStd[:, 1].mean()))
        #
        # plotFolded(np.hstack(xClump), np.hstack(yClump), cStack, 'Unregistered Clumps')
        #
        # plotFolded(np.hstack(xShifted), np.hstack(yShifted), cStack, 'Registered Clumps')
        #
        # # save shiftmaps
        # #FIXME - Getting the filename through the title is super fragile - should not use pipeline.filename (or similar) instead
        # defFile = os.path.splitext(os.path.split(self.pipeline.filename)[-1])[0] + 'MultiView.sf'
        #
        # fdialog = wx.FileDialog(None, 'Save shift field as ...',
        #     wildcard='Shift Field file (*.sf)|*.sf', style=wx.SAVE, defaultDir=nameUtils.genShiftFieldDirectoryPath(), defaultFile=defFile)
        # succ = fdialog.ShowModal()
        # if (succ == wx.ID_OK):
        #     fpath = fdialog.GetPath()
        #
        #     fid = open(fpath, 'wb')
        #     json.dump(shiftWallet, fid)
        #     fid.close()

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
        from PYME.recipes.multiview import FindClumps
        recipe = self.pipeline.recipe
        recipe.add_modules_and_execute([FindClumps(recipe, input_name=self.pipeline.selectedDataSourceKey, output_name='with_clumps',
                                     time_gap_tolerance=self.clump_gap_tolerance, radius_scale=self.clump_radius_scale,
                                     radius_offset=self.clump_radius_offset, probe_aware=True),])
        
        self.pipeline.selectDataSource('with_clumps')

    def OnMergeClumps(self, event=None):
        """

        Coalesces clusters of localization data considered to be the same molecule. See
        recipes.localizations.MergeClumps. Additionally, this function updates the colour filter after the merge, and
        updates the GUI to allow for selecting individual colour channels.

        Parameters
        ----------

            None


        Notes
        -----

        """
        from PYME.recipes.multiview import MergeClumps

        if not 'clumpIndex' in self.pipeline.keys():
            logger.debug('No clumps found - running FindClumps')
            self.OnFindClumps()

        recipe = self.pipeline.recipe

        recipe.add_modules_and_execute([MergeClumps(recipe, input_name='with_clumps', output_name='clumped'),])

        self.pipeline.selectDataSource('clumped')

        # make sure the colour filter knows about the new probe key
        #self.pipeline._process_colour()
        # refresh the Colour choice selection in the GUI
        #self.visFr.CreateFoldPanel()

    def OnMapAstigmaticZ(self, event=None, useMD = True):
        """

        Uses sigmax and sigmay values from astigmatic fits to look up a z-position using calibration curves.
        See Analysis.points.astigmatism.astigTools.lookup_astig_z

        Parameters
        ----------

            None


        Notes
        -----

        """

        from PYME.recipes.multiview import MapAstigZ
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
                                    style=wx.FD_OPEN, defaultDir=nameUtils.genShiftFieldDirectoryPath())
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
        
        # # hold off auto-running the recipe until we configure things
        # recipe.trait_set(execute_on_invalidation=False)
        # try:
        #     mapping_module = MapAstigZ(recipe, input_name=self.pipeline.selectedDataSourceKey,
        #                                astigmatism_calibration_location=pathToMap, output_name='z_mapped')
        #
        #     recipe.add_module(mapping_module)
        #     if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
        #         return
        #
        #     # FIXME - figure out why configuring just the new module doesn't give us an OK button
        #     # if not mapping_module.configure_traits(view=mapping_module.pipeline_view):
        #     #     return #handle cancel
        #     # recipe.add_module(mapping_module)
        #
        #     recipe.execute()
        # finally:  # make sure that we configure the pipeline recipe as it was
        #     recipe.trait_set(execute_on_invalidation=True)

        mapping_module = MapAstigZ(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                   astigmatism_calibration_location=pathToMap, output_name='z_mapped')

        if mapping_module.configure_traits(kind='modal'):
            recipe.add_modules_and_execute([mapping_module,])
            
            # keep a reference for debugging
            
            self._amm = mapping_module
            
            self.pipeline.selectDataSource('z_mapped')
            self.visFr.RefreshView() #TODO - is this needed?
        #self.visFr.CreateFoldPanel()
            
    def OnAstigQualityControl(self, event=None):
        import matplotlib.pyplot as plt
        import json

        with open(self._amm.astigmatism_calibration_location, 'r') as f:
            acal = json.loads(f.read())
            
        
        plt.figure()

        for i in range(4):
            plt.subplot(4, 2, 2 * i + 1)
            plt.plot(acal[i]['z'], acal[i]['sigmax'])
            plt.scatter(-self.pipeline['astigmatic_z'], self.pipeline['sigmax%d' % i], s=2, c=self.pipeline['probe'])
            plt.ylabel('sigmax%d' % i)
            plt.grid()
            plt.subplot(4, 2, 2 * i + 2)
            plt.plot(acal[i]['z'], acal[i]['sigmay'])
            plt.scatter(-self.pipeline['astigmatic_z'], self.pipeline['sigmay%d' % i], s=2, c=self.pipeline['probe'])
            plt.ylabel('sigmay%d' % i)
            plt.grid()
        

    def OnCheckAstigmatismCalibration(self, event=None):
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
    visFr.multiview = MultiviewMapper(visFr)
