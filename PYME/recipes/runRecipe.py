#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:07:57 2015

@author: david
"""
import matplotlib
from PYME.recipes import modules
from PYME.recipes import Recipe
from argparse import ArgumentParser
from PYME.IO.image import ImageStack
import pandas as pd
import tables
from PYME.IO import tabular
from PYME.IO import MetaDataHandler
from PYME.IO import unifiedIO

import logging
logger = logging.getLogger(__name__)

import warnings

import numpy as np
#import sys

def saveDataFrame(output, filename):
    """Saves a pandas dataframe, inferring the destination type based on extension"""
    warnings.warn('saveDataFrame is deprecated, use output modules instead', DeprecationWarning)
    if filename.endswith('.csv'):
        output.to_csv(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        output.to_excell(filename)
    elif filename.endswith('.hdf'):
        tabular.MappingFilter(output).to_hdf(filename)
    else:
        tabular.MappingFilter(output).to_hdf(filename + '.hdf', 'Data')

def saveTabular(output, filename):
    """Saves a pandas dataframe, inferring the destination type based on extension"""
    warnings.warn('saveTabular is deprecated, use output modules instead', DeprecationWarning)
    if filename.endswith('.csv'):
        output.toDataFrame().to_csv(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        output.toDataFrame().to_excell(filename)
    elif filename.endswith('.hdf'):
        output.to_hdf(filename)
    else:
        output.to_hdf(filename + '.hdf', 'Data')
    
def saveOutput(output, filename):
    warnings.warn('saveOutput is deprecated, use output modules instead', DeprecationWarning)
    """Save an output variable, inferring type from the file extension"""
    if isinstance(output, ImageStack):
        try:
            output.Save(filename)
        except RuntimeError:
            output.Save(filename + '.tif')
    elif isinstance(output, tabular.TabularBase):
        saveTabular(output, filename)
    elif isinstance(output, pd.DataFrame):
        saveDataFrame(output, filename)
    elif isinstance(output, matplotlib.figure.Figure):
        output.savefig(filename)
    else: #hope we can convert to a tabular format
        saveTabular(tabular.MappingFilter(output), filename)
        
def runRecipe(recipe, inputs, outputs, context={}, metadata_defaults={}):
    """Load inputs and run recipe, saving outputs.
    
    Parameters
    ----------
    recipe  : an instance of PYME.recipes.filters.ModuleCollection
    inputs  : a dictionary mapping recipe input names to filenames. These
              are loaded and inserted into the namespace before running the
              recipe.
    outputs : a dictionary mapping recipe output names to filenames. The
              corresponding members of the namespace are saved to disk
              following execution of the recipe.
    context : a dictionary used for filename subsititutions
    metadata_defaults: a dictionary (or metadata handler) specifying metadata
               entries to use if input files have incomplete metadata
        
    """
    try:
        if not isinstance(recipe, Recipe):
            # recipe is a string
            recipe = Recipe.fromYAML(recipe)
        
        #the recipe instance might be re-used - clear any previous data
        recipe.namespace.clear()

        #load any necessary inputs and populate the recipes namespace
        recipe.load_inputs(inputs, metadata_defaults=metadata_defaults)

        ### Run the recipe ###
        res = recipe.execute()

        #Save any outputs [old-style, detected using the 'out' prefix.
        for k, v in outputs.items():
            saveOutput(recipe.namespace[k],v)

        #new style output saving - using OutputModules
        recipe.save(context)
        return recipe
    except:
        logger.exception('Error running recipe')
        raise
    

def main():
    #start by finding out what recipe we're using - different recipes can have different options    
    ap = ArgumentParser(usage = 'usage: %(prog)s [options] recipe.yaml')
    ap.add_argument('recipe')
    args, remainder = ap.parse_known_args()
    
    #load the recipe
    with open(args.recipe) as f:
        s = f.read()
        
    recipe = Recipe.fromYAML(s)
    
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
    runRecipe(recipe, inputs, outputs) #TODO - fix for contexts
        
if __name__ == '__main__':
    main()
