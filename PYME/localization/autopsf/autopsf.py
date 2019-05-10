from . import roifit
from .roifit import VS

from datetime import datetime
import warnings
import numpy as np

from PYME.IO import MetaDataHandler

import six

USE_MULTIPROC = True


def _wavelength_map_zern(lamb_source, lamb_target, zerns, curMode=None, curValue=None, mode='proportional'):
    """
    Helper function used to scale zernike modes with wavelength

    """
    if curMode is None:
        return zerns * lamb_source / lamb_target
    else:
        return zerns * lamb_source / lamb_target, curMode, curValue * lamb_source / lamb_target


def _cdic(modes, values, curMode=None, curVal=None):
    d = dict(zip(modes, values))
    if curMode:
        d[curMode] = curVal
    return d


def plotSample(data, limit=100):
    from matplotlib import pyplot as plt
    images = data['data']
    
    n_channels = images[0].shape[2]
    roiWidth = int(0.5 * (data['data'].shape[1] - 1)) * 2 + 1
    count = min([images.shape[0], limit])
    nrow = int(np.sqrt(1.0 * count / n_channels / 2))
    ncol = count // nrow
    
    #        figure(figsize=(0.5*ncol, nrow*self.chCount*0.5))
    fig, ax = plt.subplots(figsize=(0.5 * ncol, nrow * n_channels * 0.5))
    img_stacked = np.hstack([np.vstack(
        [np.hstack([images[i + j][:, :, k] for k in xrange(images[i + j].shape[2])]) for i in xrange(ncol)]) for j
                             in range(0, ncol * nrow, ncol)]).T
    ax.imshow(img_stacked, interpolation='nearest', aspect='equal')#, cmap=cm.gray)
    #        imshow(np.hstack([np.vstack([np.hstack([images[i+j][:,:,k] for k in xrange(images[i+j].shape[2])]) for i in xrange(ncol)]) for j in range(0, ncol*nrow, ncol)]).T, interpolation='nearest', )#, cmap=cm.gray)
    #        axis('image')
    #        axis('off')
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_position([0, 0, 1, 1])
    ax.plot((roiWidth * ncol * np.arange(2)[None, :] * np.ones(4)[:, None]).T,
            (roiWidth * n_channels * np.arange(1, nrow)[:, None] * np.ones(2)[None, :]).T, 'y')
    ax.plot((roiWidth * np.arange(1, ncol)[:, None] * np.ones(2)[None, :]).T,
            (roiWidth * n_channels * nrow * np.arange(2)[None, :] * np.ones(19)[:, None]).T, 'y')
    
    #        colorbar()
    fig.canvas.draw()
    return img_stacked


METADATA_STRING = 'AutoPSF'


class ZernikePSFModel(object):
    def __init__(self, n_channels=1, wavelengths=[700, ], axialShift=None, splitMode=None, NA=1.45, ns=1.51,
                 apodization='sine',
                 vectorial=True, bead=None, **kwargs):
        
        self.n_channels = n_channels
        
        if not len(wavelengths) == n_channels:
            raise RuntimeError('len(wavelengths) != n_channels. Must provide a wavelength for each channel')
        
        if n_channels > 1 and axialShift is None:
            raise RuntimeError('axialShift must be defined if n_channels > 1')
        
        self.wavelengths = wavelengths
        self.axialShift = axialShift
        
        self.splitMode = splitMode
        
        self.psf_model_settings = {}
        self.psf_model_settings['vectorial'] = vectorial
        self.psf_model_settings['apodization'] = apodization
        self.psf_model_settings['NA'] = NA
        self.psf_model_settings.update(kwargs)
        
        self._symmetry_break_modes = []
        
        self.ns = ns
        
        self.bead = bead #fixme
        
        self.results = list()
    
    def _gen_meta_data(self, sourceMdh, wavelengths=None, fitBackground=False, axialShift=None):
        mdh = MetaDataHandler.NestedClassMDHandler(sourceMdh)
        
        # setting axial shift
        if self.n_channels > 1: #need axial shift if channel > 1
            if axialShift is None:
                if 'Analysis.AxialShift' in mdh.keys():
                    if 'PSF_Extraction.shift.z' in mdh.keys():
                        warnings.warn('using Analysis.AxialShift, ignoring PSF_Extraction.shift.z in mdh')
                elif 'PSF_Extraction.shift.z' in mdh.keys(): #available if mdh from extracted PSF
                    mdh['Analysis.AxialShift'] = mdh['PSF_Extraction.shift.z']
                else:
                    raise ValueError('axial shift missing')
            else:
                if 'Analysis.AxialShift' in mdh.keys():
                    warnings.warn('overwriting Analysis.AxialShift in mdh')
                if 'PSF_Extraction.shift.z' in mdh.keys(): #available if mdh from extracted PSF
                    warnings.warn('ignoring PSF_Extraction.shift.z in mdh')
                mdh['Analysis.AxialShift'] = axialShift
        elif 'Analysis.AxialShift' in mdh.keys():
            mdh['Analysis.AxialShift'] = None
            warnings.warn('Single channel data, axial shift set to None.')
        
        # setting lamb
        if wavelengths is None:
            if not (METADATA_STRING + '.lambs' in mdh.keys()):
                raise ValueError('lamb(s) missing')
        else:
            if not self.n_channels == len(wavelengths):
                raise ValueError('mismatch between number of channels and lamb(s) provided')
            else:
                mdh[METADATA_STRING + '.lambs'] = np.asarray(wavelengths)
        
        # setting whether to fit BG
        if fitBackground is None:
            if 'Analysis.FitBackground' in mdh.keys():
                mdh[METADATA_STRING + '.FitBackground'] = mdh['Analysis.FitBackground']
            else:
                raise ValueError('FitBackground missing in mdh')
        elif type(fitBackground) is bool:
            mdh[METADATA_STRING + '.FitBackground'] = fitBackground
        else:
            raise TypeError('fitBackground can only be bool or None')
        
        # setting roisize
        mdh['Analysis.ROISize'] = self.roiSize
        
        return mdh
    
    def fit(self, data, dataMdh, modes=np.arange(4, 16), maxIter=5, zern_range=np.linspace(-1.5, 1.5, 5),
            startParams=None, fitns=False, fitBackground=False, estimator=None,
            interpolator='PYME.localization.FitFactories.Interpolators.CSInterpolator', symmetry_breaking_modes=[]):
        
        self._symmetry_break_modes = symmetry_breaking_modes
        
        if (len(data['data'].shape) > 3) and not (data['data'].shape[3] == self.n_channels):
            raise RuntimeError(
                'Model is for a PSF with %d channels - expecting the same number of data channels (not %d)' % (
                self.n_channels, data['data'].shape[3]))
        
        if (len(data['data'].shape) <= 3) and not (self.n_channels == 1):
            raise RuntimeError('Model is for a PSF with %d channels, but 1-channel data provided' % self.n_channels)
        
        if not 'voxelsize.x' in dataMdh.keys():
            raise RuntimeError('Metadata must define voxelsize')
        
        if estimator is None:
            if self.n_channels == 1:
                estimator = 'PYME.localization.FitFactories.zEstimators.genericEstimator'
            else:
                estimator = 'PYME.localization.FitFactories.zEstimators.genericDualEstimator'
        elif not isinstance(estimator, six.string_types):
            estimator = estimator.__name__
        
        if not isinstance(interpolator, six.string_types):
            interpolator = interpolator.__name__
        
        self.roiSize = int(0.5 * (data['data'].shape[1] - 1))
        
        mdh = self._gen_meta_data(dataMdh, self.wavelengths, fitBackground, self.axialShift)
        
        #hide all the dirty stuff for now, fitModes is due for cleanup
        zern_out, ns_out, logs_out = self._fit_modes(data, mdh, interpolator_name=interpolator,
                                                     estimator_name=estimator, modes=modes,
                                                     fitns=fitns, maxIter=maxIter, zern_range=zern_range,
                                                     startParams=startParams)
        
        fullResult = dict()
        fullResult['modes'] = modes
        fullResult['maxIter'] = maxIter
        fullResult['zern_range'] = zern_range
        fullResult['startParams'] = startParams
        fullResult['splitMode'] = self.splitMode
        fullResult['fittedZern'] = zern_out
        fullResult['ns'] = ns_out
        fullResult['logs'] = logs_out
        
        self.results.append(fullResult)
        
        return fullResult
    
    def saveResultsToFile(self, filename):
        from six.moves import cPickle as pickle
        #        filename = basePath + "_liveExtract_" + str(int(time.time()))
        #        if path.exists(filename):
        #            numerate = 0
        #            filename_new = '{0}_{1}'.format(filename, numerate)
        #            while path.exists(filename_new):
        #                numerate += 1
        #                filename_new = '{0}_{1}'.format(filename, numerate)
        #            filename = filename_new
        
        with open(filename, 'w') as f:
            pickle.dump(self.results, f)
        
        print('success writing to {0}'.format(filename))
    
    def plot_fit_history(self, true_vals={}):
        res = self.results[-1]
        
        self._plot_fit_history(res['modes'], *res['logs'], true_vals=true_vals)
    
    def _plot_fit_history(self, modes, zern_coeff_history, test_history, true_vals=[{}, ]):
        import matplotlib.pyplot as plt
        fig_coeff_progress = plt.subplots(1, len(modes), sharey=False, figsize=(2 * len(modes), 1.5))
        #    fig_coeff_progress[0].set_size_inches(2*len(modes), 1.5)
        fig_nchi2 = plt.subplots(1, len(modes), sharey=True, figsize=(4 * len(modes), 2 * 1.5))
        #    fig_nchi2[0].set_size_inches(2*len(modes), 1.5)
        for i, j in enumerate(modes):
            fig_coeff_progress[1][i].set_xlabel('iter')
            fig_coeff_progress[1][i].set_title('mode: {0}'.format(j))
            
            fig_nchi2[1][i].set_title('mode: {0}'.format(j))
            fig_nchi2[1][i].set_ylabel('nchi2')
        
        maxIter, nCoefChans, nModes = zern_coeff_history.shape
        
        nIt = 0
        while nIt < maxIter:
            for cur_chan in range(nCoefChans):
                zern_cs = zern_coeff_history[nIt, cur_chan, :]
                x_test_vals = test_history[nIt, cur_chan, :, :, 0]
                y_test_vals = test_history[nIt, cur_chan, :, :, 1]
                
                if np.all(x_test_vals == 0):
                    break
                
                for i in range(nModes):
                    #plotting
                    fig_coeff_progress[1][i].scatter(nIt, zern_cs[i], c=nIt)
                    #                    draw()
                    line = fig_nchi2[1][i].plot(x_test_vals[i, :], y_test_vals[i, :])
                    fig_nchi2[1][i].scatter(zern_cs[i], y_test_vals[i, :].min(),
                                            c=line[0].get_color(), lw=0)
            
            nIt += 1
        
        for i, j in enumerate(modes):
            for chan_zerns in true_vals:
                tru_val = chan_zerns.get(j, None)
                if not tru_val is None:
                    fig_coeff_progress[1][i].plot([0, maxIter], [tru_val, tru_val])
                    
                    yl = fig_nchi2[1][i].get_ylim()
                    fig_nchi2[1][i].plot([tru_val, tru_val], fig_nchi2[1][i].get_ylim())
                    fig_nchi2[1][i].plot([-tru_val, -tru_val], fig_nchi2[1][i].get_ylim(), ':')
                    fig_nchi2[1][i].set_ylim(yl)
            
            #fig_coeff_progress[1][i].set_xticks([-1, 0, 1])
            fig_nchi2[1][i].set_xticks([-1, 0, 1])
        
        fig_coeff_progress[0].canvas.draw()
        fig_nchi2[0].canvas.draw()
    
    def _symmetry_break(self, mode_num, test_vals, current_value):
        if not mode_num in self._symmetry_break_modes:
            return test_vals
        
        else:
            if current_value >= 0:
                #only test a positive range
                tvm = min(test_vals.min(), 0)
                
                tvmx = test_vals.max()
                
                return (test_vals - tvm) * tvmx / (tvmx - tvm)
            else:
                #force range to be fully negative
                tvm = max(test_vals.max(), 0)
                tvmn = test_vals.min()
                return (test_vals - tvm) * tvmn / (tvmn - tvm)
    
    def _fit_modes(self, data, mdh, interpolator_name, estimator_name, modes=np.arange(4, 16), fitns=False, maxIter=4,
                   zern_range=np.linspace(-1.0, 1.0, 5), startParams=None):
        from . import roifit
        import multiprocessing as mp
        
        wavelengths = self.wavelengths
        zern_test_values = zern_range
        n_test_values = len(zern_test_values)
        
        if self.splitMode is None:
            n_data_chans = 1
            n_coeff_chans = 1
        elif self.splitMode in ['clone',
                                'mapped']: #coefficeints are the same, or scaled by wavelength, between channels
            n_data_chans = 2
            n_coeff_chans = 1
        elif self.splitMode == 'indep': # coefficients anre independant
            n_data_chans = 2
            n_coeff_chans = 2
        else:
            raise ValueError('splitMode not recognised')
        
        if not len(wavelengths) == n_data_chans:
            raise ValueError('mismatch between splitMode and len(lambs). Have to be explicit.')
        
        startTime = datetime.now()
        
        if startParams is None:
            zernike_coeffs_by_channel = np.zeros((n_coeff_chans, modes.shape[0]))
        else:
            zernike_coeffs_by_channel = startParams.copy()
        
        #testing range is centered around current params
        #TODO - why do we do this? - we use the same range for each channel anyway
        #x_range_by_chan_and_mode = zernike_coeffs_by_channel[:, :, None] + zern_test_values[None, None, :]
        
        #allocate arrays to store history
        zern_coeff_history = np.zeros(
            (maxIter, n_coeff_chans, len(modes))) #will be too large if terminated by abs(c-co).max() < 0.1
        test_history = np.zeros((maxIter, n_coeff_chans, len(modes), n_test_values, 2))
        
        #shuffle zernike fitting order
        sortorder = np.arange(len(modes))
        
        #setup worker processes
        if USE_MULTIPROC:
            nProcs = min(len(zern_range), mp.cpu_count())
            
            task_queue = mp.Queue()
            result_queue = mp.Queue()
            
            processes = [mp.Process(target=roifit.misfallB_MP,
                                    args=(task_queue, result_queue, data)) for i in range(nProcs)]
            
            for p in processes:
                p.start()
        
        def _inner_loop(cur_chan, nIt):
            if False: #if want to random shuffle order
                np.random.shuffle(sortorder)
            
            for i in sortorder: #loop over modes
                m = modes[i]
                
                x_range_cur_mode = zern_test_values + zernike_coeffs_by_channel[cur_chan, i]
                x_range_cur_mode = self._symmetry_break(m, x_range_cur_mode, zernike_coeffs_by_channel[cur_chan, i])
                
                #save the x values used for this test
                test_history[nIt, cur_chan, i, :, 0] = x_range_cur_mode
                
                #do the fit for each zernike parameterization
                y_ = np.zeros(n_test_values)
                
                z_test_ps = []
                
                for ci, x in enumerate(x_range_cur_mode):
                    #czern = [chan0_modes, chan1_modes]
                    czern = [_cdic(modes, zernike_coeffs_by_channel[cur_chan], m, x), ]
                    
                    if n_data_chans > 1:
                        if self.splitMode == "clone":
                            czern.insert(1, czern[0])
                        elif self.splitMode == "mapped": #not implemented yet, just cloned
                            czern.insert(1, _cdic(modes,
                                                  *_wavelength_map_zern(wavelengths[0], wavelengths[1],
                                                                        zernike_coeffs_by_channel[cur_chan], m, x)))
                        elif self.splitMode == 'indep':
                            czern.insert(1 - cur_chan, _cdic(modes, zernike_coeffs_by_channel[1 - cur_chan]))
                        else:
                            raise Exception('something is wrong with ch/coeff count???')
                    
                    z_test_ps.append(czern)
                
                if USE_MULTIPROC:
                    #raise NotImplementedError()
                    #processes = [mp.Process(target=roifit.misfallB_MP, args=(out_queue, ci, data, mdh, czern, interpolator.__name__, estimator.__name__), kwargs={'ns':ns}) for ci, czern in enumerate(z_test_ps)]
                    
                    for ci, czern in enumerate(z_test_ps):
                        task_queue.put((ci, (mdh, czern, interpolator_name, estimator_name),
                                        {'ns': self.ns,
                                         'wavelengths': wavelengths,
                                         'axialShift': mdh.getOrDefault('Analysis.AxialShift', None),
                                         'colourRatio': mdh.getOrDefault('Analysis.ColourRatio', None),
                                         'psf_model_settings': self.psf_model_settings}))
                    
                    for i__ in range(len(z_test_ps)):
                        ci, _r = result_queue.get()
                        y_[ci] = _r
                
                else:
                    for ci, czern in enumerate(z_test_ps):
                        y_[ci] = roifit.misfallB(data, mdh, czern, interpolator_name, estimator_name, ns=self.ns,
                                                 wavelengths=wavelengths, psf_model_settings=self.psf_model_settings,
                                                 axialShift=mdh.getOrDefault('Analysis.AxialShift', None),
                                                 colourRatio=mdh.getOrDefault('Analysis.ColourRatio', None))
                
                test_history[nIt, cur_chan, i, :, 1] = y_
                
                #fit a quadratic and find the minimum
                d = np.linalg.lstsq(
                    np.vstack(
                        [x_range_cur_mode * x_range_cur_mode, x_range_cur_mode, np.ones_like(x_range_cur_mode)]).T,
                    y_)[0]
                
                c_old = zernike_coeffs_by_channel[cur_chan, i]
                
                # check to see if solution truly is a minimum, and update zernike coefficient, clamping to our test range
                if d[0] > 0:
                    zernike_coeffs_by_channel[cur_chan, i] = np.clip(-d[1] / (2 * d[0]), x_range_cur_mode[0],
                                                                     x_range_cur_mode[-1])
                else:
                    zernike_coeffs_by_channel[cur_chan, i] = x_range_cur_mode[np.argmin(y_)]
                
                #x_range_by_chan_and_mode[cur_chan, i] = zern_test_values + zernike_coeffs_by_channel[cur_chan, i]
                
                #feedback
                print(
                'iteration {0}, ch {1}, mode {2} done, old value {3:.3f} --> new value {4:.3f}'.format(nIt, cur_chan,
                                                                                                       m, c_old,
                                                                                                       zernike_coeffs_by_channel[
                                                                                                           cur_chan, i]))
                
                print('Time elapsed: {0}'.format(datetime.now() - startTime))
            
            zern_coeff_history[nIt, cur_chan] = zernike_coeffs_by_channel[cur_chan]
        
        def _get_new_test_range(coeff_history, nIt, chan):
            if nIt > 0:
                max_change = np.max(np.abs(coeff_history[nIt, chan] - coeff_history[nIt - 1, chan]))
                zern_range_halfsize = max(float(np.clip(3.0 * max_change, 0.1, 1)), 0.5 * zern_test_values[-1])
                
                print('range update: %f, %f' % (max_change, zern_range_halfsize))
                return np.linspace(-zern_range_halfsize, zern_range_halfsize, n_test_values)
            else:
                return zern_test_values
        
        chSeq = False # if true, finish fitting one side before the other, only applicable to 'indep' splitMode
        if chSeq:
            for cur_chan in range(n_coeff_chans):
                nIt = 0
                while nIt < maxIter:
                    _inner_loop(cur_chan, nIt)
                    
                    #update the fitting range. DOES THIS ACTUALLY WORK?
                    zern_test_values = _get_new_test_range(zern_coeff_history, nIt, cur_chan)
                    
                    nIt += 1
                    
                    print(zernike_coeffs_by_channel, nIt, self.ns)
        
        else:
            nIt = 0
            while nIt < maxIter:
                for cur_chan in range(n_coeff_chans):
                    _inner_loop(cur_chan, nIt)
                
                #update the fitting range. DOES THIS ACTUALLY WORK?
                zern_test_values = _get_new_test_range(zern_coeff_history, nIt, cur_chan)
                
                nIt += 1
                
                print(zernike_coeffs_by_channel, nIt, self.ns)
        
        if USE_MULTIPROC:
            for p in processes:
                #force proceses to throw an exception and shutdown by giving them invalid data
                task_queue.put(False)
            
            for p in processes:
                p.join()
        
        return [_cdic(modes, c_) for c_ in zernike_coeffs_by_channel], self.ns, (zern_coeff_history, test_history)


