from PYME.recipes.base import register_module, ModuleBase, Filter, OutputModule
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, Either, \
    ListFloat, ListStr, Button, ToolbarButton, Array, Directory, File
import numpy as np
import pylab
from PYME.IO import tabular, MetaDataHandler
from PYME.Analysis import MetaData
from PYME.LMVis.pipeline import *
from PYME.Analysis._fithelpers import FitModelWeighted, FitModel
from PYME.Analysis.points.DeClump.pyDeClump import mergeClumpsDyeKinetics


colours = ['r', 'k', 'b', 'g']

@register_module('DyeKineticsInit')
class DyeKineticsInit(ModuleBase):
    """
    Module which sets up everything for the actual dye kinetics code.
    A workaround for the weird way recipes are run when modules are added together without an explicitly defined UI,
    as in the bakeshop

    Without this, the only UI we see is that of the first module, which is the clump finder in this case. While this isn't
    100% needed to run the script, user-customizability would be nil, making it harder to use.
    """

    inputName = Input('input')
    outputName = Output('localizations')

    clusteringRadius = Float(80)
    blinkGapTolerance = Int(2)
    blinkRadius = Float(10.0)


    minimumClusterSize = Int(2)
    minimumKeptBlinkSize = Int(1)

    minimumOffTimeInSecondsToFit = Float(0)
    maximumOffTimeInSecondsToFit = Float(5000)
    offTimeFitBinWidthInSeconds = Float(1)

    clusterColumnName = CStr('clusterID')

    onTimesColumnName = CStr('on_times')
    offTimesColumnName = CStr('off_times')




    def execute(self, namespace):
        namespace[self.outputName] = namespace[self.inputName]



@register_module('FindClumps_DK')
class FindClumps_DK(ModuleBase):
    """
    Generates tracks / clumps of single molecules based on spatial and temporal grouping. This is appropriate for
    diffraction limited objects where size, shape, or other features do not contain useful tracking information.

    One of the main uses for this module is to chain multiple observations of a single molecule together.


    """
    inputName = Input('input')
    outputName = Output('with_clumps')
    outputClumps = Output('clumps')

    timeWindow = Int(3)
    clumpRadiusScale = Float(1.0)
    clumpRadiusVariable = CStr('error_x')
    minClumpSize = Int(1)

    def execute(self, namespace):
        import PYME.Analysis.Tracking.trackUtils as trackUtils
        from PYME.IO import tabular
        meas = namespace[self.inputName]

        with_clumps, clumps = trackUtils.findTracks2(meas, self.clumpRadiusVariable, self.clumpRadiusScale,
                                                     self.timeWindow + 2, minClumpSize=self.minClumpSize)

        try:
            with_clumps.mdh = meas.mdh
        except AttributeError:
            pass

        # clumpInfo, clumps = self.Track(meas, True)
        namespace[self.outputName] = with_clumps
        namespace[self.outputClumps] = clumps

@register_module('MergeClumpsDyeKinetics')
class MergeClumpsDyeKinetics(ModuleBase):
    """Merges Localisation data depending on the indexes indentified with labelKey
    largely identical to the stock version, only difference is the addition of a maximum value aggregator

    """
    inputName = Input('with_clumps')
    outputName = Output('merged')
    labelKey = CStr('clumpIndex')

    def execute(self, namespace):

        inp = namespace[self.inputName]

        grouped = mergeClumpsDyeKinetics(inp, labelKey=self.labelKey)
        try:
            grouped.mdh = inp.mdh
        except AttributeError:
            pass

        namespace[self.outputName] = grouped


@register_module('FindBlinkStateDurations')
class FindBlinkStateDurations(ModuleBase):
    """Associates blinks with other blinks to act as a sort of clusterer, then finds the off durations between each blink
    on a per-cluster basis

    the data is organized by blinks, where every one has an associated start and end times

    so, we use some indexing tricks. The np.roll function is used to shift the on times, such that the
    0-indexed position refers to the start time of the 1-indexed blink and the end time of the 0-indexed blink

    then, we can set the final element of the on times to the last frame

    EX:
    in this example, 'start index' will refer to the frame index associated with a blink starting to fluoresce
                     'stop index' will refer to the frame index associated with the last frame a blink emitted fluorescence

        start index = [0, 8, 13, 20, 29]
        stop index = [4, 11, 14, 25, 30]

        this makes it easy to compare a blink's own start and stop times, but we want to compare the timestamp of the
        end of one blink with the beginning of another. So, np.roll gives us the following arrays:

        start index = [8, 13, 20, 29, 0]
        stop index = [4, 11, 14, 25, 30]

        the off times can be found with a simple array subtraction, then. However, we need something to compare the
        last blink to, since negative off times are nonsensical.
        So, we drop the timestamp of the final collected frame into that position. Let's say that's == 50

        **This might be wrong, may be better to just drop that blink**

        so...

        start index = [8, 13, 20, 29, 50]
        stop index = [4, 11, 14, 25, 30]

        then...

        off times = [4, 2, 6, 4, 20]

        but, because the indexes we used to find the off times are from frames where the blink was present, we need to
        subtract one frame from each, giving us our final off times:

        off times = [3, 1, 5, 3, 19]

    **USE NOTE:
        This module relies on good clustering. If clusters are incorrectly lumped together with the same index, this
        will still go through with the subtraction, finding some off times between blinks from different molecules.
        So, some off times will be incorrect and positive, which are impossible to filter out. Some off times will
        be negative, which we can filter out.
    """

    inputName = Input('filtered_localizations')
    outputName = Output('kinetics')

    onTimesColName = CStr('on_times')
    offTimesColName = CStr('off_times')

    keepEndBlinks = Bool(True)


    labelKey = CStr('dbscanClumpID')

    def execute(self, namespace):

        ds = tabular.mappingFilter(namespace[self.inputName])


        last_frame = max(ds['t2'])+1

        start_times = ds['t']
        stop_times = ds['t2']

        on_times = np.zeros_like(ds['t'])
        off_times = np.zeros_like(ds['t'])

        for i in np.unique(ds[self.labelKey]):

            clump_data_idxs = np.where(ds[self.labelKey] == i)[0]

            on_times[clump_data_idxs] = stop_times[clump_data_idxs] - start_times[clump_data_idxs] + 1

            shifted_start_times = np.roll(start_times[clump_data_idxs], -1)

            #TODO: this approach for the final off time might be wrong. OR would have to treat it differently as bleaached
            cluster_stop_times = stop_times[clump_data_idxs]

            if self.keepEndBlinks == True:
                shifted_start_times[-1] = last_frame
            elif self.keepEndBlinks == False:
                shifted_start_times[-1] = cluster_stop_times[-1] + 1

            loop_offtimes = shifted_start_times - cluster_stop_times - 1

            off_times[clump_data_idxs] = np.round(loop_offtimes).astype('float')

        try:
            mdh = namespace[self.inputName].mdh        
            ds.mdh = mdh
            ds.mdh.setEntry('bad_offtimes.number_blinks_lost', len(np.where(off_times <= 0)[0]))
            ds.mdh.setEntry('bad_offtimes.blinks_remaining', len(np.where(off_times > 0)[0]))
        except:
            pass

        #todo: filter out bad points here, no point in blowing out non-optional filtering into unique modules, the whole thing will be cleaner then


        ds.addColumn(self.onTimesColName, on_times)
        ds.addColumn(self.offTimesColName, off_times)

        namespace[self.outputName] = ds


@register_module('FitSwitchingRates')
class FitSwitchingRates(ModuleBase):
    """
    Calculates switching rates based on data which has been run through a blink finding script (trackutils.findtracks2),
    a clustering script (hdbscan), and a script which calculates all unique on and off states (findblinkstatedurations)
    """
    inputName = Input('kinetics')
    outputName = Output('kinetics_out')
    outputEmphistData = Output('emphist_data')
    onTimesColName = CStr('on_times')
    offTimesColName = CStr('off_times')

    offTimesBinSizeInSeconds = Float(2.5)

    minOffTimeSecondsToFit = Float(0.0)
    maxOffTimeSecondsToFit = Float(5000.0)

    off_init_condition_multiple = Int(1)

    exposure_time = Float(-1.)

    def execute(self, namespace):
        from PYME.Analysis.BleachProfile.kinModels import eimod

        #todo: add higher order fits

        oncol = self.onTimesColName
        offcol = self.offTimesColName

        inp = namespace[self.inputName]

        if 'mdh' in dir(inp):
            mdh = inp.mdh
            cyctime = mdh.getEntry('Camera.CycleTime')
        else:
            cyctime = self.exposure_time

        ds = tabular.mappingFilter(inp)

        off_fit_limit_index = np.round(self.maxOffTimeSecondsToFit/cyctime).astype('int')

        offtimeslimit = np.round(self.minOffTimeSecondsToFit/cyctime).astype('int')

        onbins = np.arange(1, max(ds[oncol])+1, 1)
        offbins = np.arange(offtimeslimit, off_fit_limit_index, np.round(self.offTimesBinSizeInSeconds/cyctime).astype('int'))


        n_on, bins_on = np.histogram(ds[oncol], bins=onbins)
        n_off, bins_off = np.histogram(ds[offcol], bins=offbins)

        bins_on = bins_on[:-1]
        bins_off = bins_off[:-1]

        res_off = FitModelWeighted(eimod, [n_off[0]*self.off_init_condition_multiple, 1], n_off, 1. / (np.sqrt(n_off) + 1), bins_off * cyctime)

        res_on = FitModelWeighted(eimod, [n_on[1], 1], n_on[1:], 1. / (np.sqrt(n_on[1:]) + 1), bins_on[1:] * cyctime)

        pylab.figure()

        pylab.bar(bins_on*cyctime, n_on, width=np.diff(bins_on)[0]*cyctime, alpha=0.4, fc=colours[0])
        pylab.plot(np.linspace(bins_on[0], max(bins_on), max(bins_on) * 10) * cyctime,
                   eimod(res_on[0], np.linspace(bins_on[0], max(bins_on), max(bins_on) * 10) * cyctime), colours[0],
                   lw=3)

        pylab.ylabel('Events')
        pylab.xlabel('Event Duration [s]')
        pylab.xlim((0, np.round(n_on[-1]*1.2).astype('int')))
        pylab.title('Blink Fluorescence Durations')
        pylab.figtext(.6, .8 - .05, 'tau = %3.4f' % (res_on[0][1]), size=18, color=colours[1])
        pylab.show()


        pylab.figure()

        pylab.xlim(right=np.round(self.maxOffTimeSecondsToFit/cyctime).astype('int'))
        pylab.bar(bins_off * cyctime, n_off, width=np.diff(bins_off)[0] * cyctime, alpha=0.3, fc=colours[2])
        pylab.plot(np.linspace(bins_off[0], max(bins_off), max(bins_off) * 10) * cyctime,
                   eimod(res_off[0], np.linspace(1, max(bins_off), max(bins_off) * 10) * cyctime),
                   colours[3], lw=3, alpha=.8)
        pylab.ylabel('Events')
        pylab.xlabel('Event Duration [s]')
        pylab.ylim((.9, pylab.ylim()[1]))
        pylab.title('Blink dark times')

        pylab.figtext(.6, .8 - .05, 'tau = %3.4f' % (res_off[0][1]), size=18, color=colours[1])

        pylab.show()

        powers = np.array([res_on[0][1], res_off[0][1]])
        times = np.array([cyctime, (max(ds[oncol]).astype(float)+1)*cyctime, (max(ds[offcol]).astype(float)+1)*cyctime])

        output_list = [powers, times, n_on, n_off]

        namespace[self.outputEmphistData] = output_list

@register_module('save_EmpHistJSON')
class save_EmpHistJson(ModuleBase):
    """
    Creates empirical histogram files for use with the simulator, based on the outputs from the fitting module above.
    """
    inputName = Input('emphist_data')
    outputdir = CStr('C:\\Users\\bdr25\\Desktop\\data\\JSON files')
    outputName = Output('emphist_file')

    def execute(self, namespace):
        import io
        import json
        import time

        powers, times, n_on, n_off = namespace[self.inputName]

        new_on_nums = np.zeros(len(n_on)+1)
        new_off_nums = np.zeros(len(n_off)+1)
        new_on_nums[1:] = n_on
        new_off_nums[1:] = n_off

        json_prepped_on = np.flip((new_on_nums.reshape((len(new_on_nums), 1))).astype('int'), axis=1).tolist()
        json_prepped_off = np.flip((new_off_nums.reshape((len(new_off_nums), 1))).astype('int'), axis=1).tolist()

        dk = {}
        dk['plog'] = [0, 0]
        dk['tlog'] = [0, 0]
        dk['pmin'] = [min(powers), min(powers)]
        dk['pmax'] = [max(powers), max(powers)]
        dk['tmin'] = [times[0], times[0]]
        dk['tmax'] = [times[1], times[2]]
        dk['on'] = json_prepped_on
        dk['off'] = json_prepped_off

        output_dir = self.outputdir

        try:
            to_unicode = unicode
        except NameError:
            to_unicode = str

        timestr = time.strftime("%Y%m%d-%H%M%S")

        with io.open(output_dir + '\\' + 'test_emphist' + timestr + '.json', 'w', encoding='utf8') as outfile:
            str_ = '{"dk": ' + json.dumps(dk) + '}'
            outfile.write(to_unicode(str_))

        print('Json has been created and saved to disk')


@register_module('HDBSCANClustering')
class HDBSCANClustering(ModuleBase):
    """
    Performs HDBSCAN clustering on input dictionary

    Parameters
    ----------

        minPtsForCore: The minimum size of clusters. Technically the only required parameter.
        searchRadius: Extract DBSCAN clustering based on search radius. Skipped if 0 or None.

    Notes
    -----

    See https://github.com/scikit-learn-contrib/hdbscan
    Lots of other parameters not mapped.

    """

    input_name = Input('filtered')
    columns = ListStr(['x', 'y'])
    search_radius = Float()
    min_clump_size = Int(100)
    clump_column_name = CStr('hdbscan_id')
    clump_prob_column_name = CStr('hdbscan_prob')
    clump_dbscan_column_name = CStr('dbscan_id')
    output_name = Output('hdbscan_clustered')

    def execute(self, namespace):

        inp = namespace[self.input_name]
        mapped = tabular.mappingFilter(inp)
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_clump_size)

        clusterer.fit(np.vstack([inp[k] for k in self.columns]).T)

        # Note that hdbscan gives unclustered points label of -1, and first value starts at 0.
        # shift hdbscan labels up by one to match existing convention that a clumpID of 0 corresponds to unclumped
        mapped.addColumn(str(self.clump_column_name), clusterer.labels_ + 1)
        mapped.addColumn(str(self.clump_prob_column_name), clusterer.probabilities_)

        if not self.search_radius is None and self.search_radius > 0:
            #Extract dbscan clustering from hdbscan clusterer
            dbscan = clusterer.single_linkage_tree_.get_clusters(self.search_radius, self.min_clump_size)

            # shift dbscan labels up by one to match existing convention that a clumpID of 0 corresponds to unclumped
            mapped.addColumn(str(self.clump_dbscan_column_name), dbscan + 1)

        # propogate metadata, if present
        try:
            mapped.mdh = inp.mdh
            print('testing for mdh')
        except AttributeError:
            pass

        namespace[self.output_name] = mapped
        print('finished clustering')