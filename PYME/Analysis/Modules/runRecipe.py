#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:07:57 2015

@author: david
"""
import matplotlib
from PYME.Analysis.Modules import filters
from argparse import ArgumentParser
from PYME.DSView.image import ImageStack
import pandas as pd
import numpy as np
#import sys

def loadInput(filename):
    '''Load input data from a file
    
    Currently only handles images (anything you can open in dh5view). TODO - 
    extend to other types.
    '''
    #modify this to allow for different file types - currently only supports images
    return ImageStack(filename=filename, haveGUI=False)

def saveDataFrame(output, filename):
    '''Saves a pandas dataframe, inferring the destination type based on extension'''
    if filename.endswith('.csv'):
        output.to_csv(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        output.to_excell(filename)
    else: #append a .csv
        output.to_csv(filename + '.csv')
    
def saveOutput(output, filename):
    '''Save an output variable, inferring type from the file extension'''
    if isinstance(output, ImageStack):
        try:
            output.save(filename)
        except RuntimeError:
            output.save(filename + '.tif')
    elif isinstance(output, pd.DataFrame):
        saveDataFrame(output, filename)
    elif 'toDataFrame' in dir(output):
        saveDataFrame(output.toDataFrame(), filename)
    elif isinstance(output, matplotlib.figure.Figure):
        output.savefig(filename)
    else: #hope we can convert to a data frame
        saveDataFrame(pd.DataFrame(output), filename)
        
def runRecipe(recipe, inputs, outputs):
    '''Load inputs and run recipe, saving outputs.
    
    Parameters
    ----------
      - recipe  : an instance of PYME.Analysis.Modules.filters.ModuleCollection
      - inputs  : a dictionary mapping recipe input names to filenames. These
                  are loaded and inserted into the namespace before running the
                  recipe.
      - outputs : a dictionary mapping recipe output names to filenames. The
                  corresponding members of the namespace are saved to disk
                  following execution of the recipe.
    '''
    
    #the recipe instance might be re-used - clear any previous data
    recipe.namespace.clear()
    
    #load any necessary inputs and populate the recipes namespace    
    for k, v in inputs.items():
        recipe.namespace[k] = loadInput(v)
    
    ### Run the recipe ###
    res = recipe.execute()

    #Save any outputs
    for k, v in outputs.items():
        saveOutput(recipe.namespace[k],v)    
    

def main():
    #start by finding out what recipe we're using - different recipes can have different options    
    ap = ArgumentParser(usage = 'usage: %(prog)s [options] recipe.yaml')
    ap.add_argument('recipe')
    args, remainder = ap.parse_known_args()
    
    #load the recipe
    with open(args.recipe) as f:
        s = f.read()
        
    recipe = filters.ModuleCollection.fromYAML(s)
    
    #create a new parser to parse input and output filenames
    op = ArgumentParser()
    for ip in recipe.inputs:
        op.add_argument('--%s' % ip)
        
    for ot in recipe.outputs:
        op.add_argument('--%s' % ot)
        
    args = op.parse_args(remainder)
    
    inputs = {k: getattr(args, k) for k in recipe.inputs}
    outputs = {k: getattr(args, k) for k in recipe.outputs}
    
    ##Run the recipe    
    runRecipe(recipe, inputs, outputs)
        
if __name__ == '__main__':
    main()