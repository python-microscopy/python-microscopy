# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:33:35 2014

@author: lgoo023
"""
import numpy as np
from scipy import ndimage
try:
    import Image
except ImportError:
    from PIL import Image
import os
#from PYME.DSView import View3D

def extractBlobs(image, labels, ROISize=350, scale=.005):
    objects = ndimage.find_objects(labels)
    
    ROIData = np.zeros([ROISize, ROISize, len(objects), image.shape[3]])
    
    for ind, obj in enumerate(objects):
        mask = labels[obj] == (ind+1)
    
        slx, sly, slz = obj
        for j in range(image.shape[3]):
            #print ROIData[:(slx.stop - slx.start),:(sly.stop - sly.start),ind, j].shape, image[slx, sly, slz, j].shape
            ROIData[:(slx.stop - slx.start),:(sly.stop - sly.start),ind, j] = np.minimum(np.maximum(image[slx, sly, slz, j] - image[slx, sly, slz, j].min(), 0)*mask*255/scale, 255).squeeze()
            
    return ROIData
    
def saveBlobs(blobs, dirname, fnstub):
    for i in range(blobs.shape[2]):
        im2 = blobs[:,:,i,:].squeeze()
        
        ims = np.zeros([im2.shape[0], im2.shape[1], 3], 'uint8')
        ims[:,:,:im2.shape[2]] = im2.astype('uint8')
        
        Image.fromarray(ims).save(os.path.join(dirname, fnstub + '_%d.png'%i))
        
        
    
    


    

