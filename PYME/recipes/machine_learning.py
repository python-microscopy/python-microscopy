from .base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int,  FileOrURI, Instance, WeakRef

#try:
#    from traitsui.api import View, Item, Group
#except SystemExit:
#    print('Got stupid OSX SystemExit exception - using dummy traitsui')
#    from PYME.misc.mock_traitsui import *

import numpy as np
from six.moves import xrange
from PYME.IO import unifiedIO
from scipy import ndimage
from PYME.IO.image import ImageStack

@register_module('SVMSegment')
class SVMSegment(Filter):
    classifier = FileOrURI('')
    _classifier = CStr('')
    _cf = Instance(object, allow_none=True)
    
    def _loadClassifier(self):
        from PYME.Analysis import svmSegment
        if not (self._cf and (self._classifier == self.classifier)):
            print('loading classifier')
            self._classifier = self.classifier
            with unifiedIO.local_or_temp_filename(self.classifier) as fn:
                self._cf = svmSegment.svmClassifier(filename=fn)
    
    def apply_filter(self, data, voxelsize):
        self._loadClassifier()
        
        return self._cf.classify(data.astype('f'))
    
    def completeMetadata(self, im):
        im.mdh['SVMSegment.classifier'] = self.classifier


@register_module('CNNFilter')
class CNNFilter(Filter):
    """ Use a previously trained Keras neural network to filter the data. Used for learnt de-noising and/or
    deconvolution. Runs prediction piecewise over the image with over-lapping ROIs
    and averages the prediction results.
    
    Notes
    -----
    
    Keras and either Tensorflow or Theano must be installed set up for this module to work. This is not a default
    dependency of python-microscopy as the conda-installable versions don't have GPU support.
    
    """
    model = FileOrURI('')
    step_size = Int(14)
    
    def _load_model(self):
        from keras.models import load_model
        if not getattr(self, '_model_name', None) == self.model:
            self._model_name = self.model
            with unifiedIO.local_or_temp_filename(self._model_name) as fn:
                self._model = load_model(fn)
    
    def apply_filter(self, data, voxelsize):
        self._load_model()
        
        out = np.zeros(data.shape, 'f')
        
        _, kernel_size_x, kernel_size_y, _ = self._model.input_shape
        
        scale_factor = 1./kernel_size_x*kernel_size_y/(float(self.step_size)**2)
        
        for i_x in range(0, out.shape[0] - kernel_size_x, self.step_size):
            for i_y in range(0, out.shape[1] - kernel_size_y, self.step_size):
                d_i = data[i_x:(i_x+kernel_size_x), i_y:(i_y+kernel_size_y)].reshape(1, kernel_size_x, kernel_size_y, 1)
                p = self._model.predict(d_i).reshape(kernel_size_x, kernel_size_y)
                out[i_x:(i_x+kernel_size_x), i_y:(i_y+kernel_size_y)] += scale_factor*p
        
        return out
    
    def completeMetadata(self, im):
        im.mdh['CNNFilter.model'] = self.model

class PointFeatureBase(ModuleBase):
    """
    common base class for feature extraction routines - implements normalisation and PCA routines
    """
    
    outputColumnName = CStr('features')
    columnForEachFeature = Bool(False) #if true, outputs a column for each feature - useful for visualising
    
    normalise = Bool(True) #subtract mean and divide by std. deviation
    
    PCA = Bool(True) # reduce feature dimensionality by performing PCA - TODO - should this be a separate module and be chained instead?
    PCA_components = Int(3) # 0 = same dimensionality as features
    
    def _process_features(self, data, features):
        from PYME.IO import tabular
        out = tabular.MappingFilter(data)
        out.mdh = getattr(data, 'mdh', None)
        
        if self.normalise:
            features = features - features.mean(0)[None, :]
            sd = features.std(0)
            sd[sd==0] = 1
            features = features/sd[None,:]
            
        if self.PCA:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=(self.PCA_components if self.PCA_components > 0 else None)).fit(features)
            features = pca.transform(features)
            
            out.pca = pca #save the pca object just in case we want to look at what the principle components are (this is hacky)
            
        out.addColumn(self.outputColumnName, features)
        
        if self.columnForEachFeature:
            for i in range(features.shape[1]):
                out.addColumn('feat_%d' % i, features[:,i])
                
        return out
    
        
@register_module('PointFeaturesPairwiseDist')
class PointFeaturesPairwiseDist(PointFeatureBase):
    """
    Create a feature vector for each point in a point-cloud using a histogram of it's distances to all other points
    
    """
    inputLocalisations = Input('localisations')
    outputName = Output('features')
    
    binWidth = Float(100.) # width of the bins in nm
    numBins = Int(20) #number of bins (starting at 0)
    threeD = Bool(True)

    normaliseRelativeDensity = Bool(False) # divide by the sum of all radial bins. If not performed, the first principle component will likely be average density
    
    # def execute(self, namespace):
    #     from PYME.Analysis.points import DistHist
    #     points = namespace[self.inputLocalisations]
        
    #     if self.threeD:
    #         x, y, z = points['x'], points['y'], points['z']
    #         f = np.array([DistHist.distanceHistogram3D(x[i], y[i], z[i], x, y, z, self.numBins, self.binWidth) for i in xrange(len(x))])
    #     else:
    #         x, y = points['x'], points['y']
    #         f = np.array([DistHist.distanceHistogram(x[i], y[i], x, y, self.numBins, self.binWidth) for i in xrange(len(x))])
            
        
    #     namespace[self.outputName] = self._process_features(points, f)

    def run(self, inputLocalisations):
        from PYME.Analysis.points import DistHist
        
        if self.threeD:
            x, y, z = inputLocalisations['x'], inputLocalisations['y'], inputLocalisations['z']
            f = np.array([DistHist.distanceHistogram3D(x[i], y[i], z[i], x, y, z, self.numBins, self.binWidth) for i in xrange(len(x))])
        else:
            x, y = inputLocalisations['x'], inputLocalisations['y']
            f = np.array([DistHist.distanceHistogram(x[i], y[i], x, y, self.numBins, self.binWidth) for i in xrange(len(x))])

        return self._process_features(inputLocalisations, f)



@register_module('PointFeaturesVectorial')
class PointFeaturesVectorial(PointFeatureBase):
    """
    Create a feature vector for each point in a point-cloud using a 3D histogram of it's 
    vectorial distances (polar co-ordinates) to all other points.

    Analagous to an autocorrelation for image data
    
    """
    inputLocalisations = Input('localisations')
    outputName = Output('features')
    
    radialBinWidth = Float(100.) # width of the bins in nm
    numRadialBins = Int(20) #number of bins (starting at 0)
    numAngleBins = Int(20) #number of angular bins (theta, phi)
    reducePhi = Bool(False) # reduce phi by taking mean(), max(), std()
    reduceTheta = Bool(False) # reduce phi by taking mean(), max(), std()
    #threeD = Bool(True)

    normaliseRelativeDensity = Bool(False) # divide by the sum of all radial bins. If not performed, the first principle component will likely be average density
    
    def _reduce_features(self, f):
        if self.reducePhi:
            f = np.concatenate([f.mean(3, keepdims=True), f.std(3, keepdims=True), f.max(3, keepdims=True)], axis=3)
        
        if self.reduceTheta:
            f = np.concatenate([f.mean(2, keepdims=True), f.std(2, keepdims=True), f.max(2, keepdims=True)], axis=2)

        
        return f.reshape([f.shape[0], -1])
    
    # def execute(self, namespace):
    #     from PYME.Analysis.points.features import metal
    #     points = namespace[self.inputLocalisations]
        
    #     #if self.threeD:
    #     x, y, z = points['x'], points['y'], points['z']
    #     #f = np.array([self._reduce_features(DistHist.vectDistanceHistogram3D(x[i], y[i], z[i], x, y, z, self.numRadialBins, self.radialBinWidth, self.numAngleBins)) for i in xrange(len(x))])
    #     f = metal.Backend().vector_features_3d(x, y, z, radial_bin_size=self.radialBinWidth, n_radial_bins=self.numRadialBins, n_angle_bins=self.numAngleBins)       
    #     f = self._reduce_features(f)

    #     namespace[self.outputName] = self._process_features(points, f)

    def run(self, inputLocalisations):
        from PYME.Analysis.points.features import metal

        x, y, z = inputLocalisations['x'], inputLocalisations['y'], inputLocalisations['z']
        #f = np.array([self._reduce_features(DistHist.vectDistanceHistogram3D(x[i], y[i], z[i], x, y, z, self.numRadialBins, self.radialBinWidth, self.numAngleBins)) for i in xrange(len(x))])
        f = metal.Backend().vector_features_3d(x, y, z, radial_bin_size=self.radialBinWidth, n_radial_bins=self.numRadialBins, n_angle_bins=self.numAngleBins)       
        f = self._reduce_features(f)

        return self._process_features(inputLocalisations, f)

    
@register_module('AnnotatePoints')
class AnnotatePoints(ModuleBase):
    """
    Applies the labels from a an annotation list (currently PYME.DSView.modules.annotation.AnnotationList)
    to a tabular data set. The dataset **must** have x and y keys.

    """
    inputLocalisations = Input('filtered_localisations')
    inputAnnotations = Input('annotations')

    labelColumnName = CStr('labels')

    outputName = Output('labeled')


    # def execute(self, namespace):
    #     from PYME.IO.tabular import MappingFilter
    #     from PYME.IO import MetaDataHandler

    #     inp = namespace[self.inputLocalisations]
    #     ann = namespace[self.inputAnnotations]

    #     pts = np.array([inp['x'], inp['y']]).T
        
    #     labels = ann.label_points(pts)
    #     out = MappingFilter(inp)
    #     out.addColumn('labels', labels)
    #     out.mdh = MetaDataHandler.NestedClassMDHandler(getattr(inp, 'mdh', None))

    #     namespace[self.outputName] = out

    def run(inputLocalisations, inputAnnotations):
        from PYME.IO.tabular import MappingFilter
        from PYME.IO import MetaDataHandler

        pts = np.array([inputLocalisations['x'], inputLocalisations['y']]).T
        
        labels = inputAnnotations.label_points(pts)
        out = MappingFilter(inputLocalisations)
        out.addColumn('labels', labels)
        return out