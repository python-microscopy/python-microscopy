from .base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int,  FileOrURI

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
class svmSegment(Filter):
    classifier = FileOrURI('')
    
    def _loadClassifier(self):
        from PYME.Analysis import svmSegment
        if not (('_cf' in dir(self)) and (self._classifier == self.classifier)):
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
            features = features/features.std(0)[None,:]
            
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
    
    def execute(self, namespace):
        from PYME.Analysis.points import DistHist
        points = namespace[self.inputLocalisations]
        
        if self.threeD:
            x, y, z = points['x'], points['y'], points['z']
            f = np.array([DistHist.distanceHistogram3D(x[i], y[i], z[i], x, y, z, self.numBins, self.binWidth) for i in xrange(len(x))])
        else:
            x, y = points['x'], points['y']
            f = np.array([DistHist.distanceHistogram(x[i], y[i], x, y, self.numBins, self.binWidth) for i in xrange(len(x))])
            
        
        namespace[self.outputName] = self._process_features(points, f)
    