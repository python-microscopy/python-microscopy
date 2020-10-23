from PYME.recipes.localisations import FiducialCorrection, DBSCANClustering
from PYME.recipes.tablefilters import FilterTable, FilterTableByIDs

import wx
from PYME.IO.FileUtils import nameUtils

# def drift_correct(pipeline):
#     #pipeline=visgui.pipeline
#
#     recipe = pipeline.recipe
#     recipe.add_module(FilterTable(recipe, inputName='Fiducials',
#                                     outputName='filtered_fiducials', filters={'error_x':[0,10], 'sig': [330., 370.]}))
#
#     recipe.add_module(FiducialCorrection(recipe, inputLocalizations=pipeline.selectedDataSourceKey,
#                                          inputFiducials='filtered_fiducials',
#                                          outputName='corrected_localizations', outputFiducials='corrected_fiducials'))
#
#     recipe.execute()
#     pipeline.selectDataSource('corrected_localizations')

id_filter = None

def drift_correct(pipeline):
    global id_filter
    import numpy as np
    import matplotlib.pyplot as plt
    #pipeline=visgui.pipeline

    filters = {'error_x': [0, 10]}
    if 'sig' in pipeline.keys():
        dialog = wx.TextEntryDialog(None, 'Diameter (nm): ', 'Enter Fiducial Size', str(pipeline.mdh.getOrDefault('Analysis.FiducialSize', 1000)))

        sig = [330., 370.]

        if dialog.ShowModal() == wx.ID_OK:
            size = float(dialog.GetValue())
            sigE = float(np.sqrt((size/(np.sqrt(2)*2.35))**2 + 135.**2))  # Expected std of the bead + expected std of psf
            sig = [0.95*sigE, 1.05*sigE]

        filters['sig'] = sig

    if 'fitError_z0' in pipeline.keys():
        filters['fitError_z0'] = [0,30]

    recipe = pipeline.recipe
    filt_fiducials = FilterTable(recipe, inputName='Fiducials',
                                  outputName='filtered_fiducials', filters=filters)
    
    filt_fiducials.configure_traits(kind='modal')
    #print('Adding fiducial filter module')
    #recipe.add_module(filt_fiducials)
    
    dbs = DBSCANClustering(recipe,inputName='filtered_fiducials', outputName='clumped_fiducials', columns=['x', 'y'],
                                       searchRadius=500, minClumpSize=10, clumpColumnName='fiducialID')
    recipe.add_modules_and_execute([filt_fiducials, dbs])
    
    
    fids = recipe.namespace['clumped_fiducials']
    
    fid_ids = [int(id) for id in set(fids['fiducialID']) if id > 0]
        
    id_filter = FilterTableByIDs(recipe, inputName='clumped_fiducials', outputName='selected_fiducials',
                      idColumnName='fiducialID', ids = fid_ids)
    
    #recipe.add_module(id_filter)
    
    fc = FiducialCorrection(recipe, inputLocalizations=pipeline.selectedDataSourceKey,
                                         inputFiducials='selected_fiducials',
                                         outputName='corrected_localizations', outputFiducials='corrected_fiducials')
    
    recipe.add_modules_and_execute([id_filter, fc])
    pipeline.selectDataSource('corrected_localizations')
    

def manual_selection(pipeline):
    import matplotlib.pyplot as plt
    recipe = pipeline.recipe
    
    fids = recipe.namespace['clumped_fiducials']
    
    fid_ids = [int(id) for id in set(fids['fiducialID']) if id > 0]
    
    fid_positions = {}
    
    plt.figure()
    
    for fid_id in fid_ids:
        mask = fids['fiducialID'] == fid_id
        xc = fids['x'][mask].mean()
        yc = fids['y'][mask].mean()
        fid_positions[fid_id] = (xc, yc)
        
        plt.plot(xc, yc, 'x')
        plt.text(xc + 50, yc + 50, '%d' % fid_id)
        
        plt.axis('equal')
        plt.xlim(pipeline.imageBounds.x0, pipeline.imageBounds.x1)
        plt.ylim(pipeline.imageBounds.y0, pipeline.imageBounds.y1)

    id_filter.configure_traits()
    recipe.execute()
    
def load_fiducial_info_from_second_file(pipeline):
    msg = '''New analyses should fit fiducials at the same time as the localisations and not use this function.
    When using this function, the raw localizations of all fiducials should be loaded and not a pre-filtered version'''
    if wx.MessageBox(msg, 'Info', style=wx.OK|wx.CANCEL) != wx.OK:
        return
    
    filename = wx.FileSelector("Choose a file to open",
                               nameUtils.genResultDirectoryPath(),
                               default_extension='h5r',
                               wildcard='PYME Results Files (*.h5r)|*.h5r|Tab Formatted Text (*.txt)|*.txt|Matlab data (*.mat)|*.mat|Comma separated values (*.csv)|*.csv')
    
    #print filename
    if not filename == '':
        pipeline.OpenChannel(filename, channel_name='Fiducials')
    
    
def fiducial_diagnosis_1(pipeline):
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


def fiducial_diagnosis(pipeline):
    import numpy as np
    import matplotlib.pyplot as plt
    
    fids = pipeline.dataSources['corrected_fiducials']
    ci = fids['clumpIndex']
    
    #cis = np.arange(1, ci.max())
    
    #clump_lengths = [(ci == c).sum() for c in cis]
    
    #largest_clump = ci == cis[np.argmax(clump_lengths)]
    
    #x_c = fids['x'][largest_clump]
    #y_c = fids['y'][largest_clump]
    
    f1 = plt.figure()
    a1 = plt.axes()
    plt.title('Y residuals')
    plt.grid()
    
    f2 = plt.figure()
    plt.title('X residuals')
    a2 = plt.axes()
    plt.grid()

    f3 = plt.figure()
    plt.title('Z residuals')
    a3 = plt.axes()
    plt.grid()

    for i in range(1, ci.max()):
        mask = fids['clumpIndex'] == i
        if mask.sum() > 0:
            f_id = fids['fiducialID'][mask][0]
            
            fid_m = fids['fiducialID'] == f_id
            
            ym= fids['y'][fid_m].mean()
            xm = fids['x'][fid_m].mean()
            zm = fids['z'][fid_m].mean()
            
            a1.plot(fids['t'][mask], fids['y'][mask] - ym + f_id * 50,
                 color=plt.cm.hsv( (i % 20.0)/20.))

            a2.plot(fids['t'][mask], fids['x'][mask] - xm + f_id * 50,
                    color=plt.cm.hsv((i % 20.0) / 20.))

            a3.plot(fids['t'][mask], fids['z'][mask] - zm + f_id * 50,
                    color=plt.cm.hsv((i % 20.0) / 20.))
    
    plt.figure()
    plt.subplot(311)
    plt.plot(pipeline['t'], pipeline['driftz'])
    plt.title('Z drift from focus lock')
    plt.subplot(312)
    plt.plot(pipeline['t'], pipeline['driftx'])
    plt.title('x drift from focus lock')
    plt.subplot(313)
    plt.plot(pipeline['t'], pipeline['drifty'])
    plt.title('Y drift from focus lock')
    
    #plt.hist(x_c - x_c.mean())
    
    
def drift_auto_corr(visFr):
    from PYME.recipes import localisations
    recipe = visFr.pipeline.recipe
    
    pipeline = visFr.pipeline
    
    m = localisations.AutocorrelationDriftCorrection(recipe, inputName=pipeline.selectedDataSourceKey,
                                                     outputName='corrected_localizations')
    
    if m.configure_traits(kind='modal'):
        recipe.add_modules_and_execute([m,])
        
        pipeline.selectDataSource('corrected_localizations')
        #visFr.CreateFoldPanel() #TODO: can we capture this some other way?


def Plug(visFr):
    def correct():
        drift_correct(visFr.pipeline)
        #visFr.CreateFoldPanel()
    
    visFr.AddMenuItem('Corrections>Fiducials', 'Correct', lambda e : correct())
    visFr.AddMenuItem('Corrections>Fiducials', 'Display correction residuals', lambda e: fiducial_diagnosis(visFr.pipeline))
    visFr.AddMenuItem('Corrections>Fiducials', 'Manual selection', lambda e: manual_selection(visFr.pipeline))
    visFr.AddMenuItem('Corrections>Fiducials', 'Load fiducial fits from 2nd file', lambda e: load_fiducial_info_from_second_file(visFr.pipeline))
    visFr.AddMenuItem('Corrections', 'Autocorrelation based drift correction', lambda e : drift_auto_corr(visFr))
