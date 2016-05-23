# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:46:29 2015

@author: david
"""
import numpy as np

def readSpeckles(filename):
    with open(filename, 'r') as f:
        
        speckles = []
        currentSpeckle = None    
        
        for l in f.readlines():
            if l.startswith('#%start speckle'):
                currentSpeckle = []
                speckles.append(currentSpeckle)
            elif l.startswith('#'):
                #comment
                pass
            else:
                currentSpeckle.append([float(val) for val in l.split('\t')])
                
        return [np.array(s) for s in speckles]
            