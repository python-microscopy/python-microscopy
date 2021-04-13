# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:15:01 2015

@author: david
"""

from .base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, FileOrURI

#try:
#    from traitsui.api import View, Item, Group
#except SystemExit:
#    print('Got stupid OSX SystemExit exception - using dummy traitsui')
#    from PYME.misc.mock_traitsui import *

import numpy as np
from scipy import ndimage
from PYME.IO.image import ImageStack
from PYME.IO import tabular
from PYME.IO import MetaDataHandler

import logging
logger=logging.getLogger(__name__)

@register_module('SimpleThreshold')
class SimpleThreshold(Filter):
    threshold = Float(0.5)
    
    def applyFilter(self, data, chanNum, frNum, im):
        mask = data > self.threshold
        return mask

    def completeMetadata(self, im):
        im.mdh['Processing.SimpleThreshold'] = self.threshold
        
@register_module('FractionalThreshold') 
class FractionalThreshold(Filter):
    """Chose a threshold such that the given fraction of the total labelling is
    included in the mask.
    """
    fractionThreshold = Float(0.5)

    def applyFilter(self, data, chanNum, frNum, im):
        N, bins = np.histogram(data, bins=5000)
        #calculate bin centres
        bin_mids = (bins[:-1] )
        cN = np.cumsum(N*bin_mids)
        i = np.argmin(abs(cN - cN[-1]*(1-self.fractionThreshold)))
        threshold = bins[i]

        mask = data > threshold
        return mask

    def completeMetadata(self, im):
        im.mdh['Processing.FractionalThreshold'] = self.fractionThreshold
        
@register_module('Threshold')
class Threshold(Filter):
    """ catch all class for automatic thresholding
    
    """
    
    method = Enum(['isodata', 'otsu'])
    n_histogram_bins = Int(255)
    bin_spacing = Enum(['linear', 'log', 'adaptive'])
    
    def applyFilter(self, data, chanNum, frNum, im):
        from PYME.Analysis import thresholding
        
        if self.method == 'isodata':
            threshold = thresholding.isodata_f(data, nbins=self.n_histogram_bins, bin_spacing=self.bin_spacing)
        elif self.method =='otsu':
            threshold = thresholding.otsu(data, nbins=self.n_histogram_bins, bin_spacing=self.bin_spacing)

        mask = data > threshold
        return mask
    
 
@register_module('Label')        
class Label(Filter):
    """Asigns a unique integer label to each contiguous region in the input mask.
    Optionally throws away all regions which are smaller than a cutoff size.
    """
    minRegionPixels = Int(10)
    
    def applyFilter(self, data, chanNum, frNum, im):
        mask = data > 0.5
        labs, nlabs = ndimage.label(mask)
        
        rSize = self.minRegionPixels
        
        if rSize > 1:
            m2 = 0*mask
            objs = ndimage.find_objects(labs)
            for i, o in enumerate(objs):
                r = labs[o] == i+1
                #print r.shape
                if r.sum() > rSize:
                    m2[o] += r
                                
            labs, nlabs = ndimage.label(m2 > 0)
            
        return labs

    def completeMetadata(self, im):
        im.mdh['Labelling.MinSize'] = self.minRegionPixels
        
@register_module('SelectLabel') 
class SelectLabel(Filter):
    """Creates a mask corresponding to all pixels with the given label"""
    label = Int(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        mask = (data == self.label)
        return mask

    def completeMetadata(self, im):
        im.mdh['Processing.SelectedLabel'] = self.label

@register_module('LocalMaxima')         
class LocalMaxima(Filter):
    threshold = Float(.3)
    minDistance = Int(10)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.feature
        im = data.astype('f')/data.max()
        return skimage.feature.peak_local_max(im, threshold_abs = self.threshold, min_distance = self.minDistance, indices=False)

    def completeMetadata(self, im):
        im.mdh['LocalMaxima.threshold'] = self.threshold
        im.mdh['LocalMaxima.minDistance'] = self.minDistance
        
        
# from PYME.IO.DataSources import BaseDataSource
# class _OpticFlowDataSource(BaseDataSource.BaseDataSource):
#     def __init__(self, data, filterRadius, supportRadius, regularizationLambda):
#         self.data = data
#         self.filterRadius = filterRadius
#         self.supportRadius = supportRadius
#         self.regularizationLambda = regularizationLambda
#
#         self.additionalDims = data.additionalDims
#         self.sizeC = data.sizeC
#
#     def _calc_frame_flow(self, data, i, chanNum):
#
#
#     def getSlice(self, ind):
#         """Return the nth 2D slice of the DataSource where the higher dimensions
#         have been flattened.
#
#         equivalent to indexing contiguous 4D data with data[:,:,ind%data.shape[2], ind/data.shape[3]]
#
#         e.g. for a 100x100x50x2 DataSource, getSlice(20) would return data[:,:,20,0].squeeze()
#         whereas getSlice(75) would return data[:,:,25, 1].squeeze()
#         """
#
#         from PYME.Analysis import optic_flow
#         dx, dy = 0, 0
#
#         print('OF %d' % ind)
#
#         if ind >= 1:
#             dx, dy = optic_flow.reg_of(self.data.getSlice(ind-1).squeeze(), self.data.getSlice(ind).squeeze(),
#                                        self.filterRadius, self.supportRadius, self.regularizationLambda)
#         if (ind < (self.data.getNumSlices() - 1)):
#             dx_, dy_ = optic_flow.reg_of(self.data.getSlice(ind).squeeze(), self.data.getSlice(ind+1).squeeze(),
#                                          self.filterRadius, self.supportRadius, self.regularizationLambda)
#             dx = dx + dx_
#             dy = dy + dy_
#
#     def getSliceShape(self):
#         """Return the 2D shape of a slice"""
#         return self.data.getSliceShape()
#
#     def getNumSlices(self):
#         """Return the number of 2D slices. This is the product of the
#         dimensions > 2
#         """
#         raise self.data.getNumSlices

@register_module('OpticalFlow')
class OpticalFlow(ModuleBase):
    filterRadius = Float(1)
    supportRadius = Float(10) 
    regularizationLambda = Float(0)
    inputName = Input('input')
    outputNameX = Output('flow_x')
    outputNameY = Output('flow_y')
    
    def _calc_frame_flow(self, data, i, chanNum):
        from PYME.Analysis import optic_flow
        dx, dy = 0, 0
    
        print('OF %d' % i)
    
        if i >= 1:
            dx, dy = optic_flow.reg_of(data[:, :, i - 1, chanNum].squeeze(), data[:, :, i, chanNum].squeeze(),
                                       self.filterRadius, self.supportRadius, self.regularizationLambda)
        if (i < (data.shape[2] - 1)):
            dx_, dy_ = optic_flow.reg_of(data[:, :, i, chanNum].squeeze(), data[:, :, i + 1, chanNum].squeeze(),
                                         self.filterRadius, self.supportRadius, self.regularizationLambda)
            dx = dx + dx_
            dy = dy + dy_
            
        return dx, dy
    
    def calc_flow(self, data, chanNum):
        
        flow_x = []
        flow_y = []
        
        for i in range(0, data.shape[2]):
            dx, dy = self._calc_frame_flow(data, i, chanNum)

            flow_x.append(np.atleast_3d(dx))
            flow_y.append(np.atleast_3d(dy))                
        
        
        return np.concatenate(flow_x, 2),np.concatenate(flow_y, 2)
    
    def _mp_calc_frame_flow(self, data_s, flow_x, flow_y, frames):
        for i in frames:
            dx, dy = self._calc_frame_flow(data_s, i, 0)
        
            flow_x[:, :, i] = dx
            flow_y[:, :, i] = dy
        

    def calc_flow_mp(self, data, chanNum):
        from PYME.util.shmarray import shmarray
        import multiprocessing
        
        data_s = shmarray.zeros(list(data.shape[:3] + [1,]))
        #print data_s.shape, data.shape
        data_s[:,:,:,0] = data[:,:,:,chanNum]
        
        flow_x = shmarray.zeros(data.shape[:3])
        flow_y = shmarray.zeros(data.shape[:3])
        
        nCPUs = multiprocessing.cpu_count()
        
        all_frames = range(0, data.shape[2])
        tasks = [all_frames[i::nCPUs] for i in range(nCPUs)]

        processes = [multiprocessing.Process(target=self._mp_calc_frame_flow, args=(data_s, flow_x, flow_y, frames))
                     for frames in tasks]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
    
    
        return flow_x, flow_y
        
    def execute(self, namespace):
        import multiprocessing
        image = namespace[self.inputName]
        flow_x = []
        flow_y = []
        for chanNum in range(image.data.shape[3]):
            if False:#(image.data.shape[2] > 10) and not multiprocessing.current_process().daemon:
                #use multiple processes for computation
                fx, fy = self.calc_flow_mp(image.data, chanNum)
            else:
                fx, fy = self.calc_flow(image.data, chanNum)
                
            flow_x.append(fx)
            flow_y.append(fy)
        
        im = ImageStack(flow_x, titleStub = self.outputNameX)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        self.completeMetadata(im)
        namespace[self.outputNameX] = im
        
        im = ImageStack(flow_y, titleStub = self.outputNameY)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        self.completeMetadata(im)
        namespace[self.outputNameY] = im
        
    def completeMetadata(self, im):
        im.mdh['OpticalFlow.filterRadius'] = self.filterRadius
        im.mdh['OpticalFlow.supportRadius'] = self.supportRadius
        
@register_module('WavefrontDetection')
class WavefrontDetection(ModuleBase):
    """ Detect Ca2+ wavefronts by looking at the difference images between two consecutive frames.
    
    Wavefront position corresponds to the position of the transient maximum (i.e. the zero-crossing in the temporal
    derivative), detected by finding all pixels where the magnitude of the temporal derivative is less than
    `gradientThreshold`. An intensity threshold is also used to reject areas of the image where temporal changes are due
    to noise alone.
    
    Works best on low-pass filtered data.
    """
    inputName = Input('input')
    intensityThreshold = Float(50)
    gradientThreshold = Float(0.5)
    outputName = Output('wavefronts')
    
    def execute(self, namespace):
        from skimage.morphology import skeletonize
        img = namespace[self.inputName]
        
        data = img.data[:,:,:]
        
        out = np.zeros_like(data)
        for i in range(1, data.shape[2]):
            frnt_i = (np.abs(data[:,:,i] - data[:,:,(i-1)]) < self.gradientThreshold)*(data[:,:,i] > self.intensityThreshold)
            out[:,:,i] = skeletonize(frnt_i.squeeze())
            
        im = ImageStack(out, titleStub=self.outputName)
        im.mdh.copyEntriesFrom(img.mdh)
        im.mdh['Parent'] = img.filename
        
        namespace[self.outputName] = im
        
@register_module('WavefrontVelocity')
class WavefrontVelocity(ModuleBase):
    """
    Calculates wavefront velocity given a wavefront image and optic flow images
    """
    inputWavefronts = Input('wavefronts')
    inputFlowX = Input('flow_x')
    inputFlowY = Input('flow_y')
    timeWindow = Int(5)
    outputName = Output('wavefront_velocities')
        
        
    
    def execute(self, namespace):
        from skimage.measure import profile_line
        print('Calculating wavefront velocities')
        wavefronts = namespace[self.inputWavefronts]
        
        waves = wavefronts.data
        flow_x = namespace[self.inputFlowX].data
        flow_y = namespace[self.inputFlowY].data
        
        velocities = np.zeros(waves.shape, 'f')
        
        wave_coords = []
        #precompute arrays of wavefront coordinates
        for i in range(waves.shape[2]):
            if waves[:, :, i].max() > 0:
                xp, yp = np.argwhere(waves[:, :, i].squeeze()).T
        
                xf = flow_x[:, :, i][xp, yp]
                yf = flow_y[:, :, i][xp, yp]
        
                flow_m = np.sqrt(xf * xf + yf * yf)
                xf = xf / flow_m
                yf = yf / flow_m
                
                wave_coords.append([xp, yp, xf, yf])
            else:
                wave_coords.append([np.empty(0), np.empty(0), np.empty(0), np.empty(0)])
                
        for i in range(waves.shape[2]):
            xp, yp, xf, yf = wave_coords[i]
            print('WaveV: %d' % i)
            #print(len(xp), xp, waves[:,:,i].max())
            if len(xp) >0:
                j_vals = range(max(i - self.timeWindow, 0), min(i + self.timeWindow + 1, waves.shape[2]))
                A = np.vstack([j_vals, np.ones_like(j_vals)]).T
                #A = np.ones_like(xp)[:,None,None]*A[None,:,:]
                
                ks = np.zeros([len(xp), len(j_vals)])
                j0 = j_vals[0]
                I = np.arange(len(xp))
                for j in j_vals:
                    xp_j, yp_j, xf_j, yf_j = wave_coords[j]
                    if len(xp_j) > 0:
                        k = xf*(-xp[:,None] + xp_j[None,:]) + yf*(-yp[:,None] + yp_j[None,:])
                        km = np.sqrt(k*k)
                        #print k
                        #print km.argmin(1)
                        #print k[I,km.argmin(1)]
                        ks[:, j-j0] = k[I,km.argmin(1)]
                    else:
                        ks[:, j - j0] = np.nan #mask out our matrix for the missing data
           
                #print ks
                vels = np.zeros_like(xp, 'f')
                for k in range(len(xp)):
                    kk = ks[k,:]
                    #print kk
                    vels[k] = np.linalg.lstsq(A[~np.isnan(kk),:], kk[~np.isnan(kk)])[0][0]
                
                #print vels.shape, velocities[xp,yp,i].shape
                velocities[xp, yp, i] = vels[:,None]
                
        
        
        # for i in range(waves.shape[2]):
        #     print(i)
        #     if waves[:,:,i].max() > 0:
        #         xp, yp = np.argwhere(waves[:,:,i].squeeze()).T
        #
        #         xf = flow_x[:,:, i][xp,yp]
        #         yf = flow_y[:,:, i][xp,yp]
        #
        #         flow_m = np.sqrt(xf*xf + yf*yf)
        #         xf = xf/flow_m
        #         yf = yf/flow_m
        #
        #         j_vals= range(max(i-self.timeWindow, 0), min(i+ self.timeWindow + 1, waves.shape[2]))
        #         A = np.vstack([j_vals, np.ones_like(j_vals)]).T
        #
        #         for x_k, y_k, xf_k, yf_k in zip(xp, yp, xf, yf):
        #             prof = []
        #             start, end = (x_k - 50 * xf_k, y_k - 50 * yf_k), (x_k + 50 * xf_k, y_k + 50 * yf_k)
        #             for j in j_vals:
        #                 prof.append(np.argmax(profile_line(waves[:,:,j], start, end)))
        #
        #             prof = np.array(prof, 'f')
        #
        #             #print j_vals, prof
        #             m, c = np.linalg.lstsq(A[prof>0, :], prof[prof>0])[0]
        #
        #             velocities[x_k,y_k,i] = m

        im = ImageStack(velocities, titleStub=self.outputName)
        im.mdh.copyEntriesFrom(wavefronts.mdh)
        im.mdh['Parent'] = wavefronts.filename

        namespace[self.outputName] = im
                

class CaWave(object):
    default_recipe = '''
    - processing.OpticalFlow:
        filterRadius: 10.0
        inputName: intensity
        outputNameX: flow_x
        outputNameY: flow_y
        regularizationLambda: 0.1
        supportRadius: 30.0
    - filters.GaussianFilter:
        inputName: flow_x
        outputName: flow_xf
        processFramesIndividually: false
        sigmaX: 1.0
        sigmaY: 1.0
        sigmaZ: 5.0
    - filters.GaussianFilter:
        inputName: flow_y
        outputName: flow_yf
        processFramesIndividually: false
        sigmaX: 1.0
        sigmaY: 1.0
        sigmaZ: 5.0
    - processing.WavefrontVelocity:
        inputFlowX: flow_xf
        inputFlowY: flow_yf
        inputWavefronts: wavefronts
        outputName: wavefront_velocities
        timeWindow: 5
    - measurement.ImageHistogram:
        inputImage: wavefront_velocities
        inputMask: wavefronts
        left: 0.0
        nbins: 50
        outputName: velocity_histogram
        right: 16.0
        normalize: True
    - processing.VectorfieldAngle:
        inputX: flow_xf
        inputY: flow_yf
        inputZ: ''
        outputPhi: phi
        outputTheta: theta
    - measurement.ImageHistogram:
        inputImage: theta
        inputMask: wavefronts
        left: -3.15
        nbins: 120
        outputName: angle_hist
        right: 3.15
        normalize: True
    '''
    def __init__(self, wavefronts, intensity, trange, recipe=''):
        from PYME.recipes.base import ModuleCollection
        
        self.trange = trange
        
        if recipe == '':
            recipe = self.default_recipe
        
        self._mc = ModuleCollection.fromYAML(recipe)
        
        print('Executing wave sub-recipe')
        
        self._mc.execute(wavefronts=wavefronts, intensity=intensity)
        
        print('wave sub-recipe done')
        
    @property
    def start_frame(self):
        return int(self.trange[0])

    @property
    def end_frame(self):
        return int(self.trange[1])
        
        
    @property
    def direction_plot(self):
        import matplotlib.pyplot as plt
        import mpld3
        import warnings
        if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
            #mpld3 has messed with warnings - undo
            warnings.filters.pop(0)
        
        plt.ioff()
        f = plt.figure(figsize=(4, 3))
    
        bins = self._mc.namespace['angle_hist']['bins']
        counts = self._mc.namespace['angle_hist']['counts']
        
        plt.polar(bins, counts)
    
        plt.title('Propagation direction')
    
        plt.tight_layout(pad=2)
    
        plt.ion()
    
        ret =  mpld3.fig_to_html(f)
        
        plt.close(f)
        
        return ret
        
    
    @property
    def direction_data(self):
        import json
    
        return json.dumps(np.array([self._mc.namespace['angle_hist']['bins'],
                                    self._mc.namespace['angle_hist']['counts']]).T.tolist())
        
    @property
    def velocity_plot(self):
        import matplotlib.pyplot as plt
        import mpld3
        import warnings
        if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
            #mpld3 has messed with warnings - undo
            warnings.filters.pop(0)
    
        plt.ioff()
        f = plt.figure(figsize=(4, 3))
    
        bins = self._mc.namespace['velocity_histogram']['bins']
        counts = self._mc.namespace['velocity_histogram']['counts']
    
        plt.bar(bins, counts, width=(bins[1] - bins[0]))
        plt.xlabel('Velocity [pixels/frame]')
        plt.ylabel('Frequency')
        plt.title('Velocity distribution')
    
        plt.tight_layout(pad=2)
    
        plt.ion()

        ret = mpld3.fig_to_html(f)

        plt.close(f)

        return ret
    
    @property
    def velocity_data(self):
        import json
        
        return json.dumps(np.array([self._mc.namespace['velocity_histogram']['bins'], self._mc.namespace['velocity_histogram']['counts']]).T.tolist())
    
    @property
    def wavefront_image(self):
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        try:
            from PIL import Image
        except ImportError:
            import Image
        
        wavefronts = self._mc.namespace['wavefronts'].data
        
        nFrames = wavefronts.getNumSlices()
        
        sx, sy = wavefronts.getSliceShape()
        
        out = np.zeros([sx, sy, 3])
        
        for i in range(nFrames):
            c = np.array(plt.cm.jet(float(i)/float(nFrames))[:3])
            #print c
            #print out.shape, wavefronts.getSlice(i)[:,:,None].shape, c.shape
            out += wavefronts.getSlice(i)[:,:,None]*c[None, None, :]
            
        
        outf = BytesIO()
        
        Image.fromarray((255*out).astype('uint8')).save(outf, 'PNG')
        
        s =  outf.getvalue()
        
        outf.close()
        return s
            
    
    @property
    def velocity_image(self):
        import matplotlib.pyplot as plt
        from io import BytesIO
    
        try:
            from PIL import Image
        except ImportError:
            import Image
    
        wavefronts = self._mc.namespace['wavefronts'].data
        velocities = self._mc.namespace['wavefront_velocities'].data
        v_max =  float(velocities[:,:,:].max())
    
        nFrames = wavefronts.getNumSlices()
    
        sx, sy = wavefronts.getSliceShape()
    
        out = np.zeros([sx, sy, 3])
    
        for i in range(nFrames):
            out += wavefronts.getSlice(i)[:, :, None] * plt.cm.jet(velocities.getSlice(i)/v_max)[:,:,:3]
    
        outf = BytesIO()
    
        Image.fromarray((255 * out).astype('uint8')).save(outf, 'PNG')
    
        s = outf.getvalue()
    
        outf.close()
        return s
        

@register_module('FindCaWaves')
class FindCaWaves(ModuleBase):
    '''
    Finds contiguous calcium wave events from detected wavefronts.
    '''
    inputWavefronts = Input('wavefronts')
    inputIntensity = Input('intensity')
    
    waveRecipeFileName = CStr('')
    
    minWaveFrames = Int(5)
    minActivePixels = Int(10)
    
    outputName = Output('waves')
    
    def execute(self, namespace):
        from scipy import ndimage
        from PYME.IO.DataSources import CropDataSource
        print('Finding Ca Waves ...')
        wavefronts = namespace[self.inputWavefronts] #segmented wavefront mask
        intensity = namespace[self.inputIntensity]
        wavefront_I = np.array([wavefronts.data.getSlice(i).sum() for i in range(wavefronts.data.getNumSlices())]).squeeze()
        
        
        
        #a wave is a contiguous region of non-zero wavefronts
        wave_labels, nWaves = ndimage.label(wavefront_I > float(self.minActivePixels))
        
        print('Detected %d wave candidates' % nWaves)
        
        waves = []
        
        #print(wave_labels, nWaves)
        
        for i in range(nWaves):
            wv_idx = np.argwhere(wave_labels == (i+1))
            
            print('wave%d: wave at %d-%d' % (i, wv_idx[0], wv_idx[-1]))
            
            if len(wv_idx) >= self.minWaveFrames:
                
                trange = (wv_idx[0], wv_idx[-1])
                cropped_wavefronts = ImageStack(CropDataSource.DataSource(wavefronts.data, trange=trange),
                                                mdh=getattr(wavefronts, 'mdh', None))
                cropped_intensity = ImageStack(CropDataSource.DataSource(intensity.data, trange=trange),
                                                mdh=getattr(intensity, 'mdh', None))
                waves.append(CaWave(cropped_wavefronts, cropped_intensity, trange))
                
        
        namespace[self.outputName] = waves
                
        
        
        
            
        
        
@register_module('Gradient')         
class Gradient2D(ModuleBase):
    """
    Calculate the gradient along x and y for each channel of an ImageStack

    Parameters
    ----------
    inputName : PYME.IO.image.ImageStack
        input image
    units : Enum
        specify whether to return gradient in units of intensity/pixel or
        intensity/um. Note that intensity/um will account for anisotropic 
        voxels, while the per pixel in intensity/pixel can be direction 
        dependent.
    """
    inputName = Input('input')
    outputNameX = Output('grad_x')
    outputNameY = Output('grad_y')
    units = Enum(['intensity/pixel', 'intensity/um'])
    
    def calc_grad(self, data, chanNum):
        grad_x = []
        grad_y = []
        
        for i in range(0, data.shape[2]):
            dx, dy = np.gradient(data[:,:,i, chanNum].squeeze())
            grad_x.append(np.atleast_3d(dx))
            grad_y.append(np.atleast_3d(dy))                
        
        
        return np.concatenate(grad_x, 2),np.concatenate(grad_y, 2) 
        
    def execute(self, namespace):
        image = namespace[self.inputName]
        grad_x = []
        grad_y = []
        for chanNum in range(image.data.shape[3]):
            fx, fy = self.calc_grad(image.data, chanNum)
            if self.units == 'intensity/um':
                fx /= (image.voxelsize_nm.x / 1e3)  # [data/pix] -> [data/um]
                fy /= (image.voxelsize_nm.y / 1e3)
            grad_x.append(fx)
            grad_y.append(fy)
        
        im = ImageStack(grad_x, titleStub = self.outputNameX)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        #self.completeMetadata(im)
        namespace[self.outputNameX] = im
        
        im = ImageStack(grad_y, titleStub = self.outputNameY)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        #self.completeMetadata(im)
        namespace[self.outputNameY] = im


@register_module('Gradient3D')
class Gradient3D(ModuleBase):
    """
    Calculate the gradient along x, y, and z for each channel of an ImageStack

    Parameters
    ----------
    inputName : PYME.IO.image.ImageStack
        input image
    units : Enum
        specify whether to return gradient in units of intensity/pixel or
        intensity/um. Note that intensity/um will account for anisotropic 
        voxels, while the per pixel in intensity/pixel can be direction 
        dependent.
    """
    inputName = Input('input')
    outputNameX = Output('grad_x')
    outputNameY = Output('grad_y')
    outputNameZ = Output('grad_z')
    units = Enum(['intensity/pixel', 'intensity/um'])

    def calc_grad(self, data, chanNum):
        dx, dy, dz = np.gradient(np.atleast_3d(data[:,:,:,chanNum].squeeze()))

        return dx, dy, dz

    def execute(self, namespace):
        image = namespace[self.inputName]
        grad_x = []
        grad_y = []
        grad_z = []

        for chanNum in range(image.data.shape[3]):
            fx, fy, fz = self.calc_grad(image.data, chanNum)
            if self.units == 'intensity/um':
                fx /= (image.voxelsize_nm.x / 1e3)  # [data/pix] -> [data/um]
                fy /= (image.voxelsize_nm.y / 1e3)
                fz /= (image.voxelsize_nm.z / 1e3)
            grad_x.append(fx)
            grad_y.append(fy)
            grad_z.append(fz)

        im = ImageStack(grad_x, titleStub=self.outputNameX)
        im.mdh.copyEntriesFrom(image.mdh)
        namespace[self.outputNameX] = im

        im = ImageStack(grad_y, titleStub=self.outputNameY)
        im.mdh.copyEntriesFrom(image.mdh)
        namespace[self.outputNameY] = im

        im = ImageStack(grad_z, titleStub=self.outputNameY)
        im.mdh.copyEntriesFrom(image.mdh)
        namespace[self.outputNameZ] = im

@register_module('DirectionToMask3D')
class DirectionToMask3D(ModuleBase):
    """
    Estimates the direction from a pixel to the edge of a mask.
    """
    inputName = Input('input')
    outputNameX = Output('grad_x')
    outputNameY = Output('grad_y')
    outputNameZ = Output('grad_z')

    kernelSize = Int(7)

    def calc_grad(self, data, chanNum):
        from scipy import ndimage

        data = np.atleast_3d(data[:,:,:,chanNum].squeeze())

        ks = float(self.kernelSize)
        X,Y,Z = np.mgrid[-ks:(ks+1), -ks:(ks+1), -ks:(ks+1)]
        R = np.sqrt(X*X + Y*Y + Z*Z)

        kernel_norm = 1.0/R
        kernel_norm[ks,ks,ks] = 0

        kernel_x = X/(R*R)
        kernel_x[ks, ks, ks] = 0

        kernel_y = Y / (R * R)
        kernel_y[ks, ks, ks] = 0

        kernel_z = Z / (R * R)
        kernel_z[ks, ks, ks] = 0

        norm = np.maximum(0.01, ndimage.convolve(data, kernel_norm))

        dx = ndimage.convolve(data, kernel_x)/norm
        dy = ndimage.convolve(data, kernel_y) / norm
        dz = ndimage.convolve(data, kernel_z) / norm

        norm2 = np.maximum(.01, np.sqrt(dx*dx + dy*dy + dz*dz))

        return dx/norm2, dy/norm2, dz/norm2

    def execute(self, namespace):
        image = namespace[self.inputName]
        grad_x = []
        grad_y = []
        grad_z = []

        for chanNum in range(image.data.shape[3]):
            fx, fy, fz = self.calc_grad(image.data, chanNum)
            grad_x.append(fx)
            grad_y.append(fy)
            grad_z.append(fz)

        im = ImageStack(grad_x, titleStub=self.outputNameX)
        im.mdh.copyEntriesFrom(image.mdh)
        namespace[self.outputNameX] = im

        im = ImageStack(grad_y, titleStub=self.outputNameY)
        im.mdh.copyEntriesFrom(image.mdh)
        namespace[self.outputNameY] = im

        im = ImageStack(grad_z, titleStub=self.outputNameY)
        im.mdh.copyEntriesFrom(image.mdh)
        namespace[self.outputNameZ] = im

@register_module('VectorfieldCurl')
class VectorfieldCurl(ModuleBase):
    """Calculates the curl of a vector field defined by three inputs.


    Notes
    -----

    returns
    .. math::

        (\frac{\del F_z}{\del y} - \frac{\del F_y}{\del z}, \frac{\del F_x}{\del z} - \frac{\del F_z}{\del x}, \frac{\del F_y}{\del x} - \frac{\del F_x}{\del y})$$
    """
    inputX = Input('inp_x')
    inputY = Input('inp_y')
    inputZ = Input('inp_z')

    outputX = Output('out_x')
    outputY = Output('out_y')
    outputZ = Output('out_z')


    def execute(self, namespace):
        Fx = namespace[self.inputX].data[:,:,:,0].squeeze()
        Fy = namespace[self.inputY].data[:, :, :, 0].squeeze()
        Fz = namespace[self.inputZ].data[:, :, :, 0].squeeze()

        mdh = namespace[self.inputX].mdh

        dFzdx, dFzdy, dFzdz = np.gradient(Fz)
        dFydx, dFydy, dFydz = np.gradient(Fy)
        dFxdx, dFxdy, dFxdz = np.gradient(Fx)

        im = ImageStack(dFzdy - dFydz, titleStub=self.outputX)
        im.mdh.copyEntriesFrom(mdh)
        namespace[self.outputX] = im

        im = ImageStack(dFxdz - dFzdx, titleStub=self.outputY)
        im.mdh.copyEntriesFrom(mdh)
        namespace[self.outputY] = im

        im = ImageStack(dFydx - dFzdy, titleStub=self.outputZ)
        im.mdh.copyEntriesFrom(mdh)
        namespace[self.outputZ] = im

@register_module('VectorfieldNorm')
class VectorfieldNorm(ModuleBase):
    """Calculates the norm of a vector field defined by three inputs.


    Notes
    -----

    returns
    .. math::

        sqrt(x*x + y*y + z*z)

    Also works for 2D vector fields if inputZ is an empty string.
    """
    inputX = Input('inp_x')
    inputY = Input('inp_y')
    inputZ = Input('inp_z')

    outputName = Output('output')

    def execute(self, namespace):
        x = namespace[self.inputX].data[:,:,:,0].squeeze()
        y = namespace[self.inputY].data[:, :, :, 0].squeeze()
        if self.inputZ == '':
            z = 0
        else:
            z = namespace[self.inputZ].data[:, :, :, 0].squeeze()

        mdh = namespace[self.inputX].mdh

        norm = np.sqrt(x*x + y*y + z*z)

        im = ImageStack(norm, titleStub=self.outputName)
        im.mdh.copyEntriesFrom(mdh)
        namespace[self.outputName] = im
        
@register_module('VectorfieldAngle')
class VectorfieldAngle(ModuleBase):
    """Calculates the angle of a vector field.
    
    Theta is the angle in the x-y plane, and phi is the dip angle


    Notes
    -----

    returns
    .. math::

        sqrt(x*x + y*y + z*z)

    Also works for 2D vector fields if inputZ is an empty string.
    """
    inputX = Input('inp_x')
    inputY = Input('inp_y')
    inputZ = Input('inp_z')

    outputTheta = Output('theta')
    outputPhi = Output('phi')

    def execute(self, namespace):
        x = namespace[self.inputX].data[:,:,:,0].squeeze()
        y = namespace[self.inputY].data[:, :, :, 0].squeeze()
        
        theta = np.angle(x + 1j*y)
        
        if self.inputZ == '':
            z = 0
            phi = 0*theta
        else:
            z = namespace[self.inputZ].data[:, :, :, 0].squeeze()
            
            r = np.sqrt(x*x + y*y)
            
            phi = np.angle(r + 1j*z)

        mdh = namespace[self.inputX].mdh

        

        im = ImageStack(theta, titleStub=self.outputTheta)
        im.mdh.copyEntriesFrom(mdh)
        namespace[self.outputTheta] = im

        im = ImageStack(phi, titleStub=self.outputPhi)
        im.mdh.copyEntriesFrom(mdh)
        namespace[self.outputPhi] = im


@register_module('ProjectOnVector')         
class ProjectOnVector(ModuleBase):
    """Project onto a set of direction vectors, producing p and s components"""
    inputX = Input('inputX')
    inputY = Input('inputY')
    inputDirX = Input('dirX')
    inputDirY = Input('dirY')
    
    outputNameP = Output('proj_p')
    outputNameS = Output('proj_s')
    
    def do_proj(self, inpX, inpY, dirX, dirY):
        """project onto basis vectors"""
        norm = np.sqrt(dirX*dirX + dirY*dirY)
        dx, dy = dirX/norm, dirY/norm
        
        projX = inpX*dx + inpY*dy
        projY = -inpX*dy + inpY*dx
        
        return projX, projY      
    
    def calc_proj(self, inpX, inpY, dirX, dirY, chanNum):
        proj_p = []
        proj_s = []
        
        for i in range(0, inpX.shape[2]):
            pp, ps = self.do_proj(inpX[:,:,i, chanNum].squeeze(), inpY[:,:,i, chanNum].squeeze(),
                                  dirX[:,:,i, chanNum].squeeze(), dirY[:,:,i, chanNum].squeeze())
            proj_p.append(np.atleast_3d(pp))
            proj_s.append(np.atleast_3d(ps))                
        
        
        return np.concatenate(proj_p, 2),np.concatenate(proj_s, 2) 
        
    def execute(self, namespace):
        inpX = namespace[self.inputX]
        inpY = namespace[self.inputY]
        dirX = namespace[self.inputDirX]
        dirY = namespace[self.inputDirY]
        
        proj_p = []
        proj_s = []
        for chanNum in range(inpX.data.shape[3]):
            fx, fy = self.calc_proj(inpX.data, inpY.data, dirX.data, dirY.data, chanNum)
            proj_p.append(fx)
            proj_s.append(fy)
        
        im = ImageStack(proj_p, titleStub = self.outputNameP)
        im.mdh.copyEntriesFrom(inpX.mdh)
        im.mdh['Parent'] = inpX.filename
        
        #self.completeMetadata(im)
        namespace[self.outputNameP] = im
        
        im = ImageStack(proj_s, titleStub = self.outputNameS)
        im.mdh.copyEntriesFrom(inpX.mdh)
        im.mdh['Parent'] = inpX.filename
        
        #self.completeMetadata(im)
        namespace[self.outputNameS] = im
        

class PSFFile(FileOrURI):
    '''Custom trait that verifies that the file can be loaded as a PSF'''
    
    info_text = 'a file name for a pyme PSF (.tif or .psf)'
    
    def validate(self, object, name, value):
        value = FileOrURI.validate(self, object, name, value)
        
        # Traitsui hangs up if a file doesn't validate correctly and doesn't allow selecting a replacement - disable validation for now :(
        # FIXME
        return value
        
        if value == '':
            return value
        
        try:
            assert(value.endswith('.tif') or value.endswith('.psf')) # is the file a valid psf format?
            
            # try loading as a PSF
            object.GetPSF((70., 70., 200.), psfFilename=value)
            return value
        except Exception as e:
            import traceback
            traceback.print_exc()
            
        self.error(object, name, value)
        
        

@register_module('Deconvolve')         
class Deconvolve(Filter):
    offset = Float(0)
    method = Enum('Richardson-Lucy', 'ICTM') 
    iterations = Int(10)
    psfType = Enum('file', 'bead', 'Lorentzian', 'Gaussian')
    psfFilename = PSFFile('', exists=True) #only used for psfType == 'file'
    lorentzianFWHM = Float(50.) #only used for psfType == 'Lorentzian'
    gaussianFWHM = Float(50.) #only used for psfType == 'Lorentzian'
    beadDiameter = Float(200.) #only used for psfType == 'bead'
    regularisationLambda = Float(0.1) #Regularisation - ICTM only
    padding = Int(0) #how much to pad the image by (to reduce edge effects)
    zPadding = Int(0) # padding along the z axis
    
    processFramesIndividually = False # Make deconvolution 3D by default
    
    _psfCache = {}
    _decCache = {}

    def default_traits_view(self):
        from traitsui.api import View, Item, Group
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item(name='inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item(name='outputName'),
                    Item(name='processFramesIndividually', label='2D'),
                    Group(Item(name='method'),
                          Item(name='iterations'),
                          Item(name='offset'),
                          Item(name='padding'),
                          Item(name='zPadding'),
                          Item(name='regularisationLambda', visible_when='method=="ICTM"'),
                          label='Deconvolution Parameters'),
                    Group(Item(name='psfType'),
                          Item(name='psfFilename', visible_when='psfType=="file"'),
                          Item(name='lorentzianFWHM', visible_when='psfType=="Lorentzian"'),
                          Item(name='gaussianFWHM', visible_when='psfType=="Gaussian"'),
                          Item(name='beadDiameter', visible_when='psfType=="bead"'),
                          label='PSF Parameters'),
                    resizable = True,
                    buttons   = [ 'OK' ])
                

    
    def GetPSF(self, vshint, psfFilename=None):
        from PYME.IO.load_psf import load_psf
        
        if psfFilename is None:
            psfFilename = self.psfFilename
        
        psfKey = (self.psfType, psfFilename, self.lorentzianFWHM, self.gaussianFWHM, self.beadDiameter, vshint, self.processFramesIndividually)
        
        if not psfKey in self._psfCache.keys():
            if self.psfType == 'file':
                psf, vs = load_psf(psfFilename)
                psf = np.atleast_3d(psf)

                if self.processFramesIndividually and psf.shape[2] > 1:
                    raise RuntimeError('Selected 2D deconvolution but PSF is 3D')
                elif (not self.processFramesIndividually) and (psf.shape[2] == 1):
                    raise RuntimeError('Selected 3D deconvolution but PSF is 2D')
                
                vsa = np.array([vs.x, vs.y, vs.z])
                
                if not np.allclose(vshint, vsa, rtol=.03):
                    psf = ndimage.zoom(psf, vshint/vsa)
                
                self._psfCache[psfKey] = (psf, vs)        
            elif (self.psfType == 'Lorentzian'):
                from scipy import stats
                
                if not self.processFramesIndividually:
                    raise RuntimeError('Lorentzian PSF only supported for 2D deconvolution')
                
                sc = self.lorentzianFWHM/2.0
                X, Y = np.mgrid[-30.:31., -30.:31.]
                R = np.sqrt(X*X + Y*Y)
                
                if not vshint is None:
                    vx = vshint[0]
                else:
                    vx = sc/2.
                
                vs = type('vs', (object,), dict(x=vx, y=vx))
                
                psf = np.atleast_3d(stats.cauchy.pdf(vx*R, scale=sc))
                    
                self._psfCache[psfKey] = (psf/psf.sum(), vs)
                
            elif (self.psfType == 'Gaussian'):
                from scipy import stats
                
                if not self.processFramesIndividually:
                    raise RuntimeError('Gaussian PSF only supported for 2D deconvolution')
                
                sc = self.gaussianFWHM/2.35
                X, Y = np.mgrid[-30.:31., -30.:31.]
                R = np.sqrt(X*X + Y*Y)
                
                if not vshint is None:
                    vx = vshint[0]
                else:
                    vx = sc/2.
                
                vs = type('vs', (object,), dict(x=vx, y=vx))
                
                psf = np.atleast_3d(stats.norm.pdf(vx*R, scale=sc))
                    
                self._psfCache[psfKey] = (psf/psf.sum(), vs)
            elif (self.psfType == 'bead'):
                from PYME.Deconv import beadGen
                psf = beadGen.genBeadImage(self.beadDiameter/2, vshint)
                
                if self.processFramesIndividually:
                    # project our PSF if we are doing a 2D deconvolution.
                    psf=np.atleast_3d(psf.sum(2))
                
                vs = type('vs', (object,), dict(x=vshint[0], y=vshint[1]))
                
                self._psfCache[psfKey] = (psf/psf.sum(), vs)
                
                
        return self._psfCache[psfKey]
        
    def GetDec(self, dp, vshint):
        """Get a (potentially cached) deconvolution object"""
        from PYME.Deconv import dec, richardsonLucy
        decKey = (self.psfType, self.psfFilename, self.lorentzianFWHM, self.beadDiameter, vshint, dp.shape, self.method)
        
        if not decKey in self._decCache.keys():
            psf = self.GetPSF(vshint)[0]
            
            #create the right deconvolution object
            if self.method == 'ICTM':
                if self.psfType == 'bead':
                    dc = dec.dec_bead()
                else:
                    dc = dec.dec_conv()
            else:
                if self.psfType == 'bead':
                    dc = richardsonLucy.rlbead()
                else:
                    dc = richardsonLucy.dec_conv()
                    
            #resize the PSF to fit, and do any required FFT planning etc ...
            dc.psf_calc(psf, dp.shape)
            
            self._decCache[decKey] = dc
            
        return self._decCache[decKey]
            
    
    def applyFilter(self, data, chanNum, frNum, im):
        d = np.atleast_3d(data.astype('f') - self.offset)
        #vx, vy, vz = np.array(im.voxelsize)*1e-3
        
        #Pad the data (if desired)
        if self.padding > 0:
            padsize = np.array([self.padding, self.padding, self.zPadding])
            dp = np.ones(np.array(d.shape) + 2*padsize, 'f')*d.mean()
            weights = np.zeros_like(dp)
            px, py, pz = padsize

            dp[px:-px, py:-py, pz:-pz] = d
            weights[px:-px, py:-py, pz:-pz] = 1.
            weights = weights.ravel()
        else: #no padding
            dp = d
            weights = 1
            
        #psf, vs = self.GetPSF(im.voxelsize)
        
        #Get appropriate deconvolution object        
        dec = self.GetDec(dp, im.voxelsize)
        
        #run deconvolution
        res = dec.deconv(dp, self.regularisationLambda, self.iterations, weights).reshape(dec.shape)
        
        #crop away the padding
        if self.padding > 0:
            res = res[px:-px, py:-py, pz:-pz]
        
        return res

    def completeMetadata(self, im):
        im.mdh['Deconvolution.Offset'] = self.offset
        im.mdh['Deconvolution.Method'] = self.method
        im.mdh['Deconvolution.Iterations'] = self.iterations
        im.mdh['Deconvolution.PsfType'] = self.psfType
        im.mdh['Deconvolution.PSFFilename'] = self.psfFilename
        im.mdh['Deconvolution.LorentzianFWHM'] = self.lorentzianFWHM
        im.mdh['Deconvolution.BeadDiameter'] = self.beadDiameter
        im.mdh['Deconvolution.RegularisationLambda'] = self.regularisationLambda
        im.mdh['Deconvolution.Padding'] = self.padding
        im.mdh['Deconvolution.ZPadding'] = self.zPadding
        

@register_module('DeconvolveMotionCompensating')
class DeconvolveMotionCompensating(Deconvolve):
    method = Enum('Richardson-Lucy')
    processFramesIndividually = Bool(True)
    flowScale = Float(10)
    inputFlowX = Input('flow_x')
    inputFlowY = Input('flow_y')
    
    def execute(self, namespace):
        self._flow_x = namespace[self.inputFlowX]
        self._flow_y = namespace[self.inputFlowY]
        namespace[self.outputName] = self.filter(namespace[self.inputName])
    
    def GetDec(self, dp, vshint):
        """Get a (potentially cached) deconvolution object"""
        from PYME.Deconv import richardsonLucyMVM
        decKey = (self.psfType, self.psfFilename, self.lorentzianFWHM, self.beadDiameter, vshint, dp.shape, self.method)
        
        if not decKey in self._decCache.keys():
            psf = self.GetPSF(vshint)[0]
            
            #create the right deconvolution object
            if self.psfType == 'bead':
                dc = richardsonLucyMVM.rlbead()
            else:
                dc = richardsonLucyMVM.dec_conv()
            
            #resize the PSF to fit, and do any required FFT planning etc ...
            dc.psf_calc(np.atleast_3d(psf), np.atleast_3d(dp).shape)
            
            self._decCache[decKey] = dc
        
        return self._decCache[decKey]
    
    def applyFilter(self, data, chanNum, frNum, im):
        from PYME.Analysis import optic_flow
        d = np.atleast_3d(data.astype('f') - self.offset)
    
        #Pad the data (if desired)
        if False: #self.padding > 0:
            padsize = np.array([self.padding, self.padding, self.zPadding])
            dp = np.ones(np.array(d.shape) + 2 * padsize, 'f') * d.mean()
            weights = np.zeros_like(dp)
            px, py, pz = padsize
        
            dp[px:-px, py:-py, pz:-pz] = d
            weights[px:-px, py:-py, pz:-pz] = 1.
            weights = weights.ravel()
        else: #no padding
            #dp = d
            weights = 1
    
        #Get appropriate deconvolution object
        rmv = self.GetDec(d, im.voxelsize)

        #mFr = min(frNum + 2, im.data.shape[2] -1)
        #if frNum < mFr:
        #    dx, dy = optic_flow.reg_of(im.data[:,:,frNum,chanNum].squeeze().astype('f'), im.data[:,:,mFr, chanNum].squeeze().astype('f'),
        #                               self.flowFilterRadius, self.flowSupportRadius, self.flowRegularizationLambda)
        #else:
        #    dx, dy = 0,0
        
        dx = self._flow_x.data[:,:,frNum].squeeze()
        dy = self._flow_y.data[:, :, frNum].squeeze()
        
    
        #run deconvolution
        mFr = min(frNum + 5, im.data.shape[2])
        data = np.atleast_3d([im.data[:,:,i, chanNum].astype('f').squeeze() for i in range(frNum,mFr)])
        #print data.shape
        print('MC Deconvolution - frame # %d' % frNum)
        res = rmv.deconv(data,
                         self.regularisationLambda, self.iterations, bg=0, vx = -dx*self.flowScale, vy = -dy*self.flowScale).squeeze().reshape(d.shape)
    
        #crop away the padding
        if self.padding > 0:
            res = res[px:-px, py:-py, pz:-pz]
    
        return res
    
    def default_traits_view(self):
        from traitsui.api import View, Item, Group
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item(name='inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item(name='outputName'),
                    Group(Item(name='method'),
                          Item(name='iterations'),
                          Item(name='offset'),
                          Item(name='padding'),
                          Item(name='zPadding'),
                          Item(name='regularisationLambda', visible_when='method=="ICTM"'),
                          label='Deconvolution Parameters'),
                    Group(Item(name='psfType'),
                          Item(name='psfFilename', visible_when='psfType=="file"'),
                          Item(name='lorentzianFWHM', visible_when='psfType=="Lorentzian"'),
                          Item(name='gaussianFWHM', visible_when='psfType=="Gaussian"'),
                          Item(name='beadDiameter', visible_when='psfType=="bead"'),
                          label='PSF Parameters'),
                    Group(
                          Item(name='flowScale'),
                          label='Flow estimation'),
                    resizable = True,
                    buttons   = [ 'OK' ])
        
        

    
@register_module('DistanceTransform')     
class DistanceTransform(Filter):    
    def applyFilter(self, data, chanNum, frNum, im):
        mask = 1.0*(data > 0.5)
        voxelsize = np.array(im.voxelsize)[:mask.ndim]
        dt = -ndimage.distance_transform_edt(data, sampling=voxelsize)
        dt = dt + ndimage.distance_transform_edt(1 - ndimage.binary_dilation(mask), sampling=voxelsize)
        return dt

@register_module('BinaryDilation')      
class BinaryDilation(Filter):
    iterations = Int(1)
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.binary_dilation(data, selem, iterations=self.iterations)

@register_module('BinaryErosion')         
class BinaryErosion(Filter):
    iterations = Int(1)
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.binary_erosion(data, selem, iterations=self.iterations)

@register_module('BinaryFillHoles')         
class BinaryFillHoles(Filter):
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.binary_fill_holes(data, selem)
        
@register_module('GreyDilation')      
class GreyDilation(Filter):
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.grey_dilation(data, structure=selem)

@register_module('GreyErosion')         
class GreyErosion(Filter):
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.grey_erosion(data, structure=selem)
        
@register_module('WhiteTophat')         
class WhiteTophat(Filter):
    radius = Float(1)
    
    def applyFilter(self, data, chanNum, frNum, im):
        import skimage.morphology
        
        if len(data.shape) == 3: #3D
            selem = skimage.morphology.ball(self.radius)
        else:
            selem = skimage.morphology.disk(self.radius)
        return ndimage.white_tophat(data, structure=selem)


@register_module('Watershed')         
class Watershed(ModuleBase):
    """Module with one image input and one image output"""
    inputImage = Input('input')
    inputMarkers = Input('markers')
    inputMask = Input('')
    outputName = Output('watershed')
    
    processFramesIndividually = Bool(False)
    
    def filter(self, image, markers, mask=None):
        if self.processFramesIndividually:
            filt_ims = []
            for chanNum in range(image.data.shape[3]):
                if not mask is None:
                    filt_ims.append(np.concatenate([np.atleast_3d(self.applyFilter(image.data[:,:,i,chanNum].squeeze(), markers.data[:,:,i,chanNum].squeeze(), mask.data[:,:,i,chanNum].squeeze())) for i in range(image.data.shape[2])], 2))
                else:
                    filt_ims.append(np.concatenate([np.atleast_3d(self.applyFilter(image.data[:,:,i,chanNum].squeeze(), markers.data[:,:,i,chanNum].squeeze())) for i in range(image.data.shape[2])], 2))
        else:
            if not mask is None:
                filt_ims = [np.atleast_3d(self.applyFilter(image.data[:,:,:,chanNum].squeeze(), markers.data[:,:,:,chanNum].squeeze(), mask.data[:,:,:,chanNum].squeeze())) for chanNum in range(image.data.shape[3])]
            else:
                filt_ims = [np.atleast_3d(self.applyFilter(image.data[:,:,:,chanNum].squeeze(), mask.data[:,:,:,chanNum].squeeze())) for chanNum in range(image.data.shape[3])]
            
        im = ImageStack(filt_ims, titleStub = self.outputName)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        
        #self.completeMetadata(im)
        
        return im
        
    def applyFilter(self, image,markers, mask=None):
        import skimage.morphology

        img = ((image/image.max())*2**15).astype('int16')         
        
        if not mask is None:
            return skimage.morphology.watershed(img, markers.astype('int16'), mask = mask.astype('int16'))
        else:
            return skimage.morphology.watershed(img, markers.astype('int16'))
        
    def execute(self, namespace):
        image = namespace[self.inputImage]
        markers =  namespace[self.inputMarkers]
        if self.inputMask in ['', 'none', 'None']:
            namespace[self.outputName] = self.filter(image, markers)
        else:
            mask = namespace[self.inputMask]
            namespace[self.outputName] = self.filter(image, markers, mask)
            
            
@register_module('FlatfieldAndDarkCorrect')
class FlatfiledAndDarkCorrect(ModuleBase):
    inputImage = Input('input')
    flatfieldFilename = CStr('')
    darkFilename = CStr('')
    outputName = Output('corrected')
    
    def execute(self, namespace):
        from PYME.IO.DataSources import FlatFieldDataSource
        #from PYME.IO import unifiedIO
        from PYME.IO.image import ImageStack
        image = namespace[self.inputImage]
        
        if self.flatfieldFilename != '':
            flat = ImageStack(filename=self.flatfieldFilename).data[:,:,0].squeeze()
        else:
            flat = None
        
        if not self.darkFilename == '':
            dark = ImageStack(filename=self.darkFilename).data[:,:,0].squeeze()
        else:
            dark = None
        
        ffd = FlatFieldDataSource.DataSource(image.data, image.mdh, flatfield=flat, dark=dark)

        im = ImageStack(ffd, titleStub=self.outputName)
        im.mdh.copyEntriesFrom(image.mdh)
        im.mdh['Parent'] = image.filename
        namespace[self.outputName] = im

@register_module('Colocalisation')
class Colocalisation(ModuleBase):
    """
    Calculate thresholded manders and Pearsons coefficients
    
    
    """
    inputImageA = Input('chan0')
    inputMaskA = Input('mask0')
    inputImageB = Input('chan1')
    inputMaskB = Input('mask1')
    inputRoiMask = Input('')
    outputTable = Output('coloc')
    
    def execute(self, namespace):
        from PYME.Analysis.Colocalisation import correlationCoeffs
        from PYME.IO import tabular
        
        imA = namespace[self.inputImageA].data[:,:,:,0].squeeze()
        imB = namespace[self.inputImageB].data[:,:,:,0].squeeze()
        if not np.all(imB.shape == imA.shape):
            raise RuntimeError('imageB (shape=%s) not the same size as image data (shape=%s)' % (imB.shape, imA.shape))

        mA = namespace[self.inputMaskA].data[:,:,:,0].squeeze()
        if not np.all(mA.shape == imA.shape):
            raise RuntimeError('maskA (shape=%s) not the same size as image data (shape=%s)' % (mA.shape, imA.shape))
        
        mB = namespace[self.inputMaskB].data[:,:,:,0].squeeze()
        if not np.all(mB.shape == imA.shape):
            raise RuntimeError('maskB (shape=%s) not the same size as image data (shape=%s)' % (mB.shape, imA.shape))
        
        if not self.inputRoiMask == '':
            roi_mask = namespace[self.inputRoiMask].data[:,:,:,0].squeeze() > 0.5
            if not np.all(roi_mask.shape == imA.shape):
                raise RuntimeError('ROI mask (shape=%s) not the same size as image data (shape=%s)' % (roi_mask.shape, imA.shape))

        else:
            roi_mask = None

        print('Calculating Pearson and Manders coefficients ...')
        pearson = correlationCoeffs.pearson(imA, imB, roi_mask=roi_mask)
        MA, MB = correlationCoeffs.maskManders(imA, imB, mA, mB, roi_mask=roi_mask)
        
        out = tabular.DictSource({'pearson' : pearson, 'manders_A' : MA, 'manders_B' : MB})
        
        namespace[self.outputTable] = out
          


@register_module('ColocalisationEDT')
class ColocalisationEDT(ModuleBase):
    """
    Perform distance-transform based colocalisation of an image with a mask. Returns the relative
    enrichment, and the total signal contained within a given distance from a mask.
    
    Parameters
    ===========
    
    inputImage : an intensity image
    mask : a mask (usually derived from a different channel)
    inputImageB : [optional] the intensity image for the channel used to create the mask. If present, this is used to
                    assess the colocalisation of the mask channel with itself as a control.
    outputTable : table into which to save results
    
    minimumDistance, maximumDistance, binSize : float, nm the range of distances to calculate the histogram over
    
    
    Outputs
    =======
    
    outputTable : a table containing the following columns:
            'bins' : the right hand edges of the histogram bins (as suitable for the cdf plots)
            'enrichment' : the enrichment of the label at a given distance from the mask (when compared to a uniform spatial distribution)
            'enclosed' : the fraction of the signal enclosed within a given distance from the mask
            'enclosed_area' : the fraction of the total area enclosed within a given radius. This gives you the curve you
                              would see if the label was randomly distributed.
            'enrichment_m' : enrichment of the mask source channel (if provided) at a given distance from the mask. This is a control for the thresholding
            'enclosed_m' : fraction of mask source channel signal (if provided) within a given distance from the mask. This is a control for the thresholding.
                    
    
    Notes
    ======
    
    - If the input image has multiple colour channels, the 0th channel will be taken (i.e. split channels first)
    - To do colocalisation both ways between two images, you will need two copies of this module
    
    TODO: handle channel names appropriately, support for ROI masks
    """
    inputImage = Input('input')
    inputMask = Input('mask')
    inputImageB = Input('')
    inputRoiMask = Input('')
    outputTable = Output('edt_coloc')
    outputPlot = Output('coloc_plot')
    
    minimumDistanceNM = Float(-500.)
    maximumDistanceNM = Float(2000.)
    binSizeNM = Float(100.)
    
    
    def execute(self, namespace):
        from PYME.IO import tabular
        from PYME.Analysis.Colocalisation import edtColoc
        from PYME.recipes.graphing import Plot
        
        bins = np.arange(float(self.minimumDistanceNM), float(self.maximumDistanceNM), float(self.binSizeNM))

        im = namespace[self.inputImage]
        imA = im.data[:, :, :, 0].squeeze()
        voxelsize = im.voxelsize[:imA.ndim]
        
        m_im = namespace[self.inputMask]
        mask = m_im.data[:,:,:,0].squeeze() > 0.5

        if not np.all(mask.shape == imA.shape):
            raise RuntimeError('Mask (shape=%s) not the same size as image data (shape=%s)' % (mask.shape, imA.shape))
        
        if not self.inputRoiMask == '':
            roi_mask = namespace[self.inputRoiMask].data[:,:,:,0].squeeze() > 0.5
            if not np.all(roi_mask.shape == imA.shape):
                raise RuntimeError('ROI mask (shape=%s) not the same size as image data (shape=%s)' % (roi_mask.shape, imA.shape))
        else:
            roi_mask = None
        

        bins_, enrichment, enclosed, enclosed_area = edtColoc.image_enrichment_and_fraction_at_distance(imA, mask, voxelsize,
                                                                                               bins, roi_mask=roi_mask)
        
        out = tabular.MappingFilter(tabular.DictSource({'bins' : bins[1:], 'enrichment' : enrichment, 'enclosed' : enclosed, 'enclosed_area' : enclosed_area}))
        out.mdh = getattr(im, 'mdh', None)
        
        if not self.inputImageB == '':
            imB = namespace[self.inputImageB].data[:,:, :,0].squeeze()
            if not np.all(imB.shape == imA.shape):
                raise RuntimeError(
                    'ImageB (shape=%s) not the same size as image data (shape=%s)' % (imB.shape, imA.shape))
            
            bins_, enrichment_m, enclosed_m, _ = edtColoc.image_enrichment_and_fraction_at_distance(imB, mask, voxelsize,
                                                                                             bins, roi_mask=roi_mask)
            out.addColumn('enrichment_m', enrichment_m)
            out.addColumn('enclosed_m', enclosed_m)
            
        else:
            enrichment_m = None
            enclosed_m = None

        namespace[self.outputTable] = out
        namespace[self.outputPlot] = Plot(lambda: edtColoc.plot_image_dist_coloc_figure(bins, enrichment, enrichment_m,
                                                                                        enclosed, enclosed_m,
                                                                                        enclosed_area,
                                                                                        nameA=m_im.names[0],
                                                                                        nameB=im.names[0]))

        
        
@register_module('AverageFramesByZStep')
class AverageFramesByZStep(ModuleBase):
    """
    Averages frames acquired at the same z-position, as determined by the associated events, or (fall-back) metadata.

    Parameters
    ----------
    input_image : string
        name of an ImageStack instance, with metadata / events describing which frames were taken at which z-position.
    input_zvals : string
        [optional] name of a table mapping frames to z values. If empty, the image events are used.

    Returns
    -------
    output : traits.Output
        ImageStack instance, where frames taken at the same z-position have been averaged together.
    """

    input_image = Input('input')
    input_zvals = Input('') #if undefined, use events
    z_column_name = CStr('z')
    output = Output('averaged_by_frame')

    def execute(self, namespace):
        from PYME.Analysis import piezo_movement_correction
        from scipy.stats import mode
        import time

        image_stack = namespace[self.input_image]

        if self.input_zvals == '':
            # z from events
            frames = np.arange(image_stack.data.shape[2], dtype=int)

            z_vals = piezo_movement_correction.correct_target_positions(frames, image_stack.events, image_stack.mdh)
        else:
            #z values are provided as input
            z_vals = namespace[self.input_zvals][self.z_column_name]
        
        # later we will mash z with %3.3f, round here so we don't duplicate steps
        # TODO - make rounding precision a parameter?
        z_vals = np.round(z_vals, decimals=3)

        # make sure everything is sorted. We'll carry the args to sort, rather than creating another full array
        frames_z_sorted = np.argsort(z_vals)
        z = z_vals[frames_z_sorted]
        z_steps, count = np.unique(z, return_counts=True)

        n_steps = len(z_steps)
        logger.debug('Averaged stack size: %d' % n_steps)

        new_stack = []
       # TODO - should we default to zero or abort?
        t = image_stack.mdh.getOrDefault('StartTime', 0)
        fudged_events = []
        cycle_time = image_stack.mdh.getOrDefault('Camera.CycleTime', 1.0)
        for ci in range(image_stack.data.shape[3]):
            data_avg = np.zeros((image_stack.data.shape[0], image_stack.data.shape[1], n_steps))
            start = 0
            for si in range(n_steps):
                for fi in range(count[si]):
                    # sum frames from this step directly into the output array
                    data_avg[:, :, si] += image_stack.data[:, :, frames_z_sorted[start + fi], ci].squeeze()
                start += count[si]
                fudged_events.append(('ProtocolFocus', t, '%d, %3.3f' % (si, z_steps[si])))
                fudged_events.append((('StartAq', t, '%d' % si)))
                t += cycle_time
            # complete the average for this color channel and append to output
            new_stack.append(data_avg / count[None, None, :])

        # FIXME  - make this follow the correct event dtype
        fudged_events = np.array(fudged_events, dtype=[('EventName', 'S32'), ('Time', '<f8'), ('EventDescr', 'S256')])
        averaged = ImageStack(new_stack, mdh=MetaDataHandler.NestedClassMDHandler(image_stack.mdh), events=fudged_events)

        # fudge metadata, leaving breadcrumbs
        averaged.mdh['Camera.CycleTime'] = cycle_time
        averaged.mdh['StackSettings.NumSlices'] = n_steps
        averaged.mdh['StackSettings.StepSize'] = abs(mode(np.diff(z))[0][0])

        namespace[self.output] = averaged


@register_module('ResampleZ')
class ResampleZ(ModuleBase):
    """
    Resamples input stack at even intervals along z using a linear interpolation. If the input stack contains multiple
    frames taken at the same z position, the stack should first be run through AverageFramesByZStep.

    Parameters
    ----------
    input: Input
        ImageStack instance
    z_sampling: Float
        Spacing to resample the stack axially, units of micrometers

    Returns
    -------
    output: Output
        ImageStack instance

    """

    input = Input('input')
    z_sampling = Float(0.05)
    output = Output('regular_stack')

    def execute(self, namespace):
        from PYME.Analysis import piezo_movement_correction
        from scipy.interpolate import RegularGridInterpolator

        stack = namespace[self.input]

        # grab z from events if we can
        frames = np.arange(stack.data.shape[2], dtype=int)

        z_vals = piezo_movement_correction.correct_target_positions(frames, stack.events, stack.mdh)

        x = np.arange(0, stack.mdh['voxelsize.x'] * stack.data.shape[0], stack.mdh['voxelsize.x'])
        y = np.arange(0, stack.mdh['voxelsize.y'] * stack.data.shape[1], stack.mdh['voxelsize.y'])

        # generate grid for sampling
        new_z = np.arange(np.min(z_vals), np.max(z_vals), self.z_sampling)
        xx, yy, zz = np.meshgrid(x, y, new_z, indexing='ij')
        # RegularGridInterpolator needs z to be strictly ascending need to average frames from the same step first
        uni, counts = np.unique(z_vals, return_counts=True)
        if np.any(counts > 1):
            raise RuntimeError('Resampling requires one frame per z-step. Please run AverageFramesByZStep first')
        I = np.argsort(z_vals)
        sorted_z_vals = z_vals[I]
        regular = []
        for ci in range(stack.data.shape[3]):
            interp = RegularGridInterpolator((x, y, sorted_z_vals), stack.data[:, :, :, ci][:,:,I], method='linear')
            regular.append(interp((xx, yy, zz)))

        mdh = MetaDataHandler.DictMDHandler({
            'RegularizedStack': True,
            'StackSettings.StepSize': self.z_sampling,
            'StackSettings.StartPos': new_z[0],
            'StackSettings.EndPos': new_z[-1],
            'StackSettings.NumSlices': len(new_z),
            'voxelsize.z': self.z_sampling
        })
        mdh.mergeEntriesFrom(stack.mdh)
        
        regular_stack = ImageStack(regular, mdh=mdh)

        namespace[self.output] = regular_stack

@register_module('BackgroundSubtractionMovingAverage')
class BackgroundSubtractionMovingAverage(ModuleBase):
    """
    Estimates and subtracts the background of a series using a sliding window average on a per-pixel basis.

    Parameters
    ----------
    input_name : Input
        PYME.IO.ImageStack
    window = List
        Describes the window, much like range or numpy.arrange, format is [start, finish, stride]

    Returns
    -------
    output_name = Output
        PYME.IO.ImageStack of the background-subtracted 'input_name' series

    Notes
    -----

    input and output images are the same size.

    """

    input_name = Input('input')
    window = List([-32, 0, 1])
    output_name = Output('background_subtracted')

    percentile = 0

    def execute(self, namespace):
        from PYME.IO.DataSources import BGSDataSource
        from PYME.IO.image import ImageStack
        series = namespace[self.input_name]

        bgs = BGSDataSource.DataSource(series.data, bgRange=self.window)
        bgs.setBackgroundBufferPCT(self.percentile)

        background = ImageStack(data=bgs, mdh=MetaDataHandler.NestedClassMDHandler(series.mdh))

        background.mdh['Parent'] = series.filename
        background.mdh['Processing.SlidingWindowBackground.Percentile'] = self.percentile
        background.mdh['Processing.SlidingWindowBackground.Window'] = self.window

        namespace[self.output_name] = background

@register_module('BackgroundSubtractionMovingPercentile')
class BackgroundSubtractionMovingPercentile(BackgroundSubtractionMovingAverage):
    """
    Estimates the background of a series using a sliding window and taking an (adjusted) percentile
    (e.g. median at percentile = 0.5) over that window on a per-pixel basis.

    Parameters
    ----------
    input_name : Input
        PYME.IO.ImageStack
    percentile : Float
        Percentile to take as the background after sorting within the time window along each pixel
    window = List
        Describes the window, much like range or numpy.arrange, format is [start, finish, stride]

    Returns
    -------
    output_name = Output
        PYME.IO.ImageStack of the background-subtracted 'input_name' series

    Notes
    -----
    The percentile background isn't a simple percentile, but is adjusted slightly - see PYME.IO.DataSource.BGSDataSource

    input and output images are the same size.
    """
    percentile = Float(0.25)


@register_module('Projection')
class Projection(Filter):
    """ Project image along an axis
    
    TODO - make this more efficient - we currently force the whole stack into memory
    """
    
    kind = Enum(['Mean', 'Sum', 'Max', 'Median', 'Std', 'Min'])
    axis = Int(2)
    
    processFramesIndividually = False
    
    def applyFilter(self, data, chanel_num, frame_num, image):
        if self.kind == 'Mean':
            return np.mean(data, axis=int(self.axis))
        if self.kind == 'Sum':
            return np.sum(data, axis=int(self.axis))
        if self.kind == 'Max':
            return np.max(data, axis=int(self.axis))
        if self.kind == 'Median':
            return np.median(data, axis=int(self.axis))
        if self.kind == 'Std':
            return np.std(data, axis=int(self.axis))
        if self.kind == 'Min':
            return np.min(data, axis=int(self.axis))

@register_module('StatisticsByFrame')
class StatisticsByFrame(ModuleBase):
    """
    Iterates through the time/z-position dimension of an ImageStack, calculating basic statistics for each frame,
    optionally using a 2D or 3D mask in the process.
    
    NOTE: only operates on first colour channel of stack.

    Parameters
    ----------
    input_name : Input
        PYME.IO.ImageStack
    mask : Input
        PYME.IO.ImageStack. Optional mask to only calculate metrics

    Returns
    -------
    output_name = Output


    Notes
    -----

    """

    input_name = Input('input')

    mask = Input('')

    output_name = Output('cluster_metrics')

    def execute(self, namespace):
        from scipy import stats

        series = namespace[self.input_name]
        data = series.data

        if self.mask == '':
            mask = None
        else:
            # again, handle our mask being either 2D, 3D, or 4D. NB - no color (4D) handling implemented at this point
            mask = namespace[self.mask].data

        var = np.empty(data.shape[2], dtype=float)
        mean = np.empty_like(var)
        median = np.empty_like(var)
        mode = np.empty_like(var)

        for si in range(data.shape[2]):
            slice_data = data[:,:,si, 0]
            
            if mask is not None:
                if mask.shape[2] == 1:
                    slice_data = slice_data[mask[:,:,0,0].astype('bool')]
                elif mask.shape[2] == data.shape[2]:
                    slice_data = slice_data[mask[:,:,si,0].astype('bool')]
                else:
                    raise RuntimeError('Mask dimensions do not match data dimensions')

            var[si] = np.var(slice_data)
            mean[si] = np.mean(slice_data)
            median[si] = np.median(slice_data)
            mode[si] = stats.mode(slice_data, axis=None)[0][0]

        # package up and ship-out results
        res = tabular.DictSource({'variance': var, 'mean': mean, 'median': median, 'mode': mode})
        try:
            res.mdh = series.mdh
        except:
            pass
            
        namespace[self.output_name] = res

@register_module('DarkAndVarianceMap')
class DarkAndVarianceMap(ModuleBase):
    input = Input('input')
    output_variance = Output('variance')
    output_dark = Output('dark')
    dark_threshold = Float(1e4)  # this really should depend on the gain mode (12bit vs 16 bit etc)
    variance_threshold = Float(300**2)  # again this is currently picked fairly arbitrarily
    blemish_variance = Float(1e8) #set broken pixels to super high variance)
    start = Int(0)
    end = Int(-1)

    def execute(self, namespace):
        from PYME.Analysis import gen_sCMOS_maps

        image = namespace[self.input]
        
        dark_map, variance_map = gen_sCMOS_maps.generate_maps(image, self.start, self.end,
                                                              darkthreshold=self.dark_threshold,
                                                              variancethreshold=self.variance_threshold,
                                                              blemishvariance=self.blemish_variance)

        namespace[self.output_dark] = dark_map
        namespace[self.output_variance] = variance_map
