from PYME.recipes.localisations import FiducialCorrection
from PYME.recipes.tablefilters import FilterTable

def drift_correct(pipeline):
    #pipeline=visgui.pipeline

    recipe = pipeline.recipe
    recipe.add_module(FilterTable(recipe, inputName='Fiducials',
                                    outputName='filtered_fiducials', filters={'error_x':[0,10], 'sig': [330., 370.]}))

    recipe.add_module(FiducialCorrection(recipe, inputLocalizations=pipeline.selectedDataSourceKey,
                                         inputFiducials='filtered_fiducials',
                                         outputName='corrected_localizations', outputFiducials='corrected_fiducials'))
    
    recipe.execute()
    pipeline.selectDataSource('corrected_localizations')
    
    
def fiducial_diagnosis(pipeline):
    import numpy as np
    import matplotlib.pyplot as plt
    
    fids = pipeline.datasources['corrected_fiducials']
    ci = fids['clumpIndex']
    
    cis = np.arange(1, ci.max())
    
    clump_lengths =  [(ci==c).sum() for c in cis]
    
    largest_clump = ci == cis[np.argmax(clump_lengths)]
    
    x_c = fids['x'][largest_clump]
    y_c = fids['y'][largest_clump]
    
    plt.figure()
    
    plt.hist(x_c - x_c.mean())
    
    
def Plug(visFr):
    visFr.AddMenuItem('Extras>Fiducials', 'Correct', lambda e : drift_correct(visFr.pipeline))