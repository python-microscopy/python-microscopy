from .base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int,  File

#try:
#    from traitsui.api import View, Item, Group
#except SystemExit:
#    print('Got stupid OSX SystemExit exception - using dummy traitsui')
#    from PYME.misc.mock_traitsui import *

import numpy as np
from scipy import ndimage
from PYME.IO.image import ImageStack

@register_module('SVMSegment')
class svmSegment(Filter):
    classifier = File('')
    
    def _loadClassifier(self):
        from PYME.Analysis import svmSegment
        if not '_cf' in dir(self):
            self._cf = svmSegment.svmClassifier(filename=self.classifier)
    
    def applyFilter(self, data, chanNum, frNum, im):
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
    model = File('')
    step_size = Int(14)
    
    def _load_model(self):
        from keras.models import load_model
        if not getattr(self, '_model_name', None) == self.model:
            self._model_name = self.model
            self._model = load_model(self._model_name) #TODO - make cluster-aware
    
    def applyFilter(self, data, chanNum, frNum, im):
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