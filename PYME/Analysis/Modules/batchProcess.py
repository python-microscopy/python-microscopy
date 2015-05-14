#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:23:57 2015

@author: david
"""
import matplotlib
matplotlib.use('Cairo')

from PYME.Analysis.Modules import runRecipe
from PYME.Analysis.Modules import filters
import os
import glob
from argparse import ArgumentParser

import multiprocessing

NUM_PROCS = multiprocessing.cpu_count()

def runRec(args):
    print args
    runRecipe.runRecipe(*args)

def main():
    #start by finding out what recipe we're using - different recipes can have different options    
    ap = ArgumentParser()#usage = 'usage: %(prog)s [options] recipe.yaml')
    ap.add_argument('recipe')
    ap.add_argument('output_dir')
    ap.add_argument('-n', '--num-processes', default=NUM_PROCS)
    args, remainder = ap.parse_known_args()
    
    #load the recipe
    with open(args.recipe) as f:
        s = f.read()
        
    recipe = filters.ModuleCollection.fromYAML(s)

    output_dir = args.output_dir
    num_procs = args.num_processes
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #create a new parser to parse input and output filenames
    op = ArgumentParser()
    for ip in recipe.inputs:
        op.add_argument('--%s' % ip)
        
        
    args = op.parse_args(remainder)
    
    inputGlobs = {k: glob.glob(getattr(args, k)) for k in recipe.inputs}
    
    #print inputGlobs

    inputLengths = [len(v) for v in inputGlobs.values()]
    if not (min(inputLengths) == max(inputLengths)):
        raise RuntimeError('The number of entries in each input category must be equal')
    
    taskParams = []
    outputNames = recipe.outputs
    
    for i in range(inputLengths[0]):
        in_d = {k:v[i] for k, v in inputGlobs.items()}
        
        fns = os.path.join(output_dir, os.path.splitext(os.path.split(in_d.values()[0])[-1])[0])
        out_d = {k:('%s_%s'% (fns,k)) for k in  outputNames}
        
        taskParams.append((recipe, in_d, out_d))
        
    pool = multiprocessing.Pool(num_procs)
    
    pool.map(runRec, taskParams)
    
    #print taskParams
#    for tp in taskParams:
#        print tp[1]
#        
#        runRecipe.runRecipe(*tp)
        
        
if __name__ == '__main__':
    main()