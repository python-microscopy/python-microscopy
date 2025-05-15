from PYME.recipes import Recipe
from PYME.recipes import modules
from PYME.IO.image import ImageStack
from PYME import resources
import numpy as np
import os


recipe_1= '''
- filters.GaussianFilter:
    inputName: input
    outputName: filtered_image
- filters.MeanFilter:
    dimensionality: XYZ
    inputName: input
    outputName: filtered_image_1
- filters.MedianFilter:
    inputName: input
    outputName: filtered_image_2
- base.Add:
    inputName0: filtered_image
    inputName1: filtered_image_1
    outputName: filtered_image_3
    processFramesIndividually: true
- processing.SimpleThreshold:
    inputName: normed
    outputName: mask
- base.JoinChannels:
    inputChan0: filtered_image_2
    inputChan1: mask
    inputChan2: ''
    inputChan3: ''
    outputName: comp
- base.Normalize:
    inputName: filtered_image_3
    outputName: normed
- filters.Zoom:
    inputName: comp
    outputName: zoomed
    zoom: 0.5

'''

recipe_2='''
- localisations.AddPipelineDerivedVars:
    inputEvents: ''
    inputFitResults: FitResults
    outputEventMaps: event_maps
    outputLocalizations: Localizations
- localisations.ProcessColour:
    input: Localizations
    output: colour_mapped
- tablefilters.FilterTable:
    filters:
      error_x:
      - 0
      - 30
      error_y:
      - 0
      - 30
    inputName: colour_mapped
    outputName: filtered_localizations
'''

def test_recipe_1():
    rec = Recipe.fromYAML(recipe_1)
    im = ImageStack(filename=os.path.join(resources.get_test_data_dir(), 't_im.tif'))
    
    rec.execute(input=im)
    assert(np.allclose(rec.namespace['zoomed'].data_xyztc.shape, (88, 80, 241, 1, 2)))

def test_recipe_dict_trait_serialisation():
    """ Extremely minimal test to ensure we can save a recipe to YAML
    
    - protects against massive breakages in saving (ie we have something which is not yaml serialisable)
    - does NOT check that round-trip is accurate
    - chosen recipe (recipe2) should specifically test the regression in #1592 (ie modified Dict traits)
    - has really bad coverage on other recipe modules and trait types

    TODO - should we find some way of validating round-trip success (can't directly compare strings as ordering, string escaping, and indentation not garuanteed to be preserved)
    """
    
    rec = Recipe.fromYAML(recipe_2)
    yaml_str = rec.toYAML()
    assert(yaml_str != '')
