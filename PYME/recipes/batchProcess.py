#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:23:57 2015

@author: david
"""
from PYME.recipes import runRecipe
from PYME.recipes import Recipe, modules
import os
import glob
from argparse import ArgumentParser
import traceback

import multiprocessing

NUM_PROCS = multiprocessing.cpu_count()

def runRec(args):
    #print args
    try:
        import matplotlib.pyplot as plt
    
        old_backend = plt.get_backend()
        plt.switch_backend('SVG')
        
        r = runRecipe.runRecipe(*args)
        
        plt.switch_backend(old_backend)
        return r
    except Exception as e:
        traceback.print_exc()
        raise
    
def bake(recipe, inputGlobs, output_dir, num_procs = NUM_PROCS, start_callback=None, success_callback=None, error_callback=None, metadata_defaults={}):
    """Run a given recipe over using multiple proceses.
    
    Parameters
    ----------
      recipe:       The recipe to run
      inputGlobs:   A dictionary of where the keys are the input names, and each
                    entry is a list of filenames which provide the data for that
                    input.
      output_dir:   The directory to save the output in
      num_procs:    The number of worker processes to launch (defaults to the number of CPUs)
      
    """
    
    #check that we've supplied the right number of images for each named input/channel
    inputLengths = [len(v) for v in inputGlobs.values()]
    if not (min(inputLengths) == max(inputLengths)):
        raise RuntimeError('The number of entries in each input category must be equal')
        
    taskParams = []
    outputNames = recipe.outputs
    
    for i in range(inputLengths[0]):
        in_d = {k:v[i] for k, v in inputGlobs.items()}

        # FIXME - change to os.path.splitext(os.path.basename(list(in_d.values())[0]).split('?')[0])[0], otherwise file_stub = ?level=N for supertiles
        file_stub = os.path.splitext(os.path.basename(list(in_d.values())[0]))[0]
        
        fns = os.path.join(output_dir, file_stub)
        out_d = {k:('%s_%s'% (fns,k)) for k in  outputNames}

        cntxt = {'output_dir' : output_dir, 'file_stub': file_stub}

        taskParams.append((recipe.toYAML(), in_d, out_d, cntxt, dict(metadata_defaults)))

    last_rec = None

    if num_procs == 1:
        # map(runRec, taskParams)  # map now returns iterator, which means this never runs unless we convert to list
        #[runRec(task) for task in taskParams]  # skip the map and just make the list we need anyway
        for task in taskParams:
            in_d = task[1]
            if start_callback:
                start_callback(in_d)
            try:
                last_rec = runRec(task)
                if success_callback:
                    success_callback(in_d)
            except:
                if error_callback:
                    error_callback(in_d)
                raise
        return last_rec        
    else:
        pool = multiprocessing.Pool(num_procs)
    
        r = pool.map_async(runRec, taskParams, error_callback = lambda e: traceback.print_exception(e, value=e, tb=e.__traceback__))
        
        r.wait()
        pool.close()
        return None

def bake_recipe(recipe_filename, inputGlobs, output_dir, *args, **kwargs):
    with open(recipe_filename) as f:
        s = f.read()
    
    recipe = Recipe.fromYAML(s)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    bake(recipe, inputGlobs, output_dir, *args, **kwargs)
    

def main():
    # set matplotlib backend for offline figure generation #TODO - move this further down (ie. to the figure generation code itself)?
    import matplotlib
    matplotlib.use('Cairo', warn=False)
    
    #start by finding out what recipe we're using - different recipes can have different options    
    ap = ArgumentParser()#usage = 'usage: %(prog)s [options] recipe.yaml')
    ap.add_argument('recipe')
    ap.add_argument('output_dir')
    ap.add_argument('-n', '--num-processes', default=NUM_PROCS)
    args, remainder = ap.parse_known_args()
    
    #load the recipe
    with open(args.recipe) as f:
        s = f.read()
        
    recipe = Recipe.fromYAML(s)

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
    
    bake(recipe, inputGlobs, output_dir, num_procs)
        
        
if __name__ == '__main__':
    main()
