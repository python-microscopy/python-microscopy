# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:47:24 2014

@author: lgoo023
"""

import os
import sys
import glob
import numpy as np

from jinja2 import Environment, PackageLoader
env = Environment(loader=PackageLoader('PYME.experimental.RandomPresent', 'templates'))

class img(object):
    def __init__(self, fn):
        self.fn = fn
        self.cls = os.path.split(fn)[-1][0]

def genPage(dirname, trainingFraction=.25):
    imagenames = glob.glob(os.path.join(dirname, '*.png'))
    
    ri = np.random.rand(len(imagenames))
    RI = ri.argsort()
    
    imagenames = [imagenames[i] for i in RI]
    
    imgs = [img(fn) for fn in imagenames]
    
    trainingImgs = []
    testImgs = []
    
    for im in imgs:
        if np.random.rand() < trainingFraction:
            trainingImgs.append(im)
        else:
            testImgs.append(im)
            
    testTemplate = env.get_template('View.html')
    cheatTemplate = env.get_template('CheatView.html')
    
    #print imgs, trainingImgs
    
    s =  testTemplate.render(training=trainingImgs, test = testImgs)
    
    f = open(os.path.join(dirname, 'test.html'), 'w')
    f.write(s)
    f.close()
    
    s =  cheatTemplate.render(training=trainingImgs, test = testImgs)
    
    f = open(os.path.join(dirname, 'cheat.html'), 'w')
    f.write(s)
    f.close()
        
if (__name__ == '__main__'):
     genPage(sys.argv[1])
