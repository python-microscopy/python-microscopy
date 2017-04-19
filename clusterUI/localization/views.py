from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from django.core.files.uploadhandler import TemporaryFileUploadHandler
from django.views.decorators.csrf import csrf_exempt, csrf_protect
import django.forms

import logging
logger=logging.getLogger(__name__)

import collections
from PYME.localization import MetaDataEdit as mde
import PYME.localization.FitFactories
FINDING_PARAMS = [#mde.ChoiceParam('Analysis.FitModule', 'Fit module:', default='LatGaussFitFR', choices=PYME.localization.FitFactories.resFitFactories),
                  mde.FloatParam('Analysis.DetectionThreshold', 'Detection threshold:', 1.0),
                  mde.IntParam('Analysis.DebounceRadius', 'Debounce radius:', 4),
                  mde.IntParam('Analysis.StartAt', 'Start at:', default=30),
                  ]


BACKGROUND_PARAMS = [#mde.IntParam('Analysis.StartAt', 'Start at:', default=30),
              mde.RangeParam('Analysis.BGRange', 'Background range:', default=[-30, 0]),
              mde.BoolParam('Analysis.subtractBackground', 'Subtract background in fit', default=True),
              mde.BoolFloatParam('Analysis.PCTBackground', 'Use percentile for background', default=False,
                                 helpText='', ondefault=0.25, offvalue=0),
    ]

SCMOS_PARAMS = [
              mde.FilenameParam('Camera.VarianceMapID', 'Variance Map:',
                                prompt='Please select variance map to use ...', wildcard='TIFF Files|*.tif',
                                filename=''),
              mde.FilenameParam('Camera.DarkMapID', 'Dark Map:', prompt='Please select dark map to use ...',
                                wildcard='TIFF Files|*.tif', filename=''),
              mde.FilenameParam('Camera.FlatfieldMapID', 'Flatfield Map:',
                                prompt='Please select flatfield map to use ...', wildcard='TIFF Files|*.tif',
                                filename=''),
    ]

FIDUCIAL_PARAMS = [
              mde.BoolParam('Analysis.TrackFiducials', 'Track Fiducials', default=False),
              mde.FloatParam('Analysis.FiducialThreshold', 'Fiducial Threshold', default=1.8),
              ]


#class GroupedForm(django.forms.Form)

#we need to keep hold of the pusher objects so they don't get GCed whilst still pushing
#pushers = []


def settings_form(analysisModule):
    #params = FINDING_PARAMS + DEFAULT_PARAMS

    categorized_fields = collections.OrderedDict()

    categorized_fields['Detection'] = [p.formField() for p in FINDING_PARAMS]
    categorized_fields['Background'] = [p.formField() for p in BACKGROUND_PARAMS]
    categorized_fields['sCMOS'] = [p.formField() for p in SCMOS_PARAMS]
    categorized_fields['Fiducials'] = [p.formField() for p in FIDUCIAL_PARAMS]


    try:
        fm = __import__('PYME.localization.FitFactories.' + analysisModule,
                        fromlist=['PYME', 'localization', 'FitFactories'])

        categorized_fields['Module'] = [p.formField() for p in fm.PARAMETERS]

        #params += fm.PARAMETERS
    except AttributeError:
        pass

    fields = {}
    fieldnames_by_category = collections.OrderedDict()
    #for p in params:
    #    fields.update(p.formField())
    #print categorized_fields
    for k, v in categorized_fields.items():
        #print v
        fields.update(v)
        fieldnames_by_category[k] = [name for name, field in v]

    fields['fieldnames_by_category'] = fieldnames_by_category

    return type('%sForm' % analysisModule, (django.forms.Form, ), fields)


# Create your views here.
def settings(request, analysisModule='LatGaussFitFR'):
    import PYME.localization.FitFactories
    analysisModule=analysisModule.encode()

    if request.method == 'POST':
        f = settings_form(analysisModule)(request.POST)
    else:
        f = settings_form(analysisModule)()

    fields_by_category = collections.OrderedDict()
    for cat, fieldnames in f.fieldnames_by_category.items():
        fields_by_category[cat] = [f[name] for name in fieldnames]

    return render(request, 'localization/settings_form.html', {'form':f, 'analysisModule' : analysisModule,
                                                               'analysisModuleChoices' : PYME.localization.FitFactories.resFitFactories,
                                                               'categorized_fields' : fields_by_category})

def _verifyClusterResultsFilename(resultsFilename):
    from PYME.IO import clusterIO
    import os
    if clusterIO.exists(resultsFilename):
        di, fn = os.path.split(resultsFilename)
        i = 1
        stub = os.path.splitext(fn)[0]
        while clusterIO.exists(os.path.join(di, stub + '_%d.h5r' % i)):
            i += 1

        resultsFilename = os.path.join(di, stub + '_%d.h5r' % i)

    return resultsFilename

def _launch_localize(analysisMDH, seriesName):
    import logging
    import json
    from PYME.ParallelTasks import HTTPTaskPusher
    from PYME.IO import MetaDataHandler
    from PYME.Analysis import MetaData
    from PYME.IO.FileUtils.nameUtils import genClusterResultFileName
    from PYME.IO import unifiedIO

    resultsFilename = _verifyClusterResultsFilename(genClusterResultFileName(seriesName))
    logging.debug('Results file: ' + resultsFilename)

    resultsMdh = MetaDataHandler.NestedClassMDHandler()#analysisMDH)
    # NB - anything passed in analysis MDH will whipe out corresponding entries in the series metadata
    resultsMdh.update(json.loads(unifiedIO.read(seriesName + '/metadata.json')))
    resultsMdh.update(analysisMDH)

    resultsMdh['EstimatedLaserOnFrameNo'] = resultsMdh.getOrDefault('EstimatedLaserOnFrameNo', resultsMdh.getOrDefault('Analysis.StartAt', 0))
    MetaData.fixEMGain(resultsMdh)
    #resultsMdh['DataFileID'] = fileID.genDataSourceID(image.dataSource)

    #TODO - do we need to keep track of the pushers in some way (we currently rely on the fact that the pushing thread
    #will hold a reference
    pusher = HTTPTaskPusher.HTTPTaskPusher(dataSourceID=seriesName,
                                                metadata=resultsMdh, resultsFilename=resultsFilename)

    logging.debug('Queue created')


def localize(request, analysisModule='LatGaussFitFR'):
    #import json
    from PYME.IO import MetaDataHandler
    import copy
    import time
    from PYME.Analysis import MetaData

    analysisModule = request.POST.get('Analysis.FitModule', analysisModule).encode()

    f = settings_form(analysisModule)(request.POST)

    f.is_valid()

    f.cleaned_data['Analysis.FitModule'] = analysisModule

    #print json.dumps(f.cleaned_data)
    # NB - any metadata entries given here will override the series metadata later: pass analysis settings only
    analysisMDH = MetaDataHandler.NestedClassMDHandler()
    analysisMDH.update(f.cleaned_data)

    #print request.GET
    #print request.POST.getlist('series', [])

    #resultsFilename = _verifyResultsFilename(genResultFileName(image.seriesName))
    
    remaining_series = request.POST.getlist('series', [])
    
    nAttempts = 0
    
    while len(remaining_series) > 0 and nAttempts < 3:
        nAttempts += 1
        
        seriesToLaunch = copy.copy(remaining_series)
        remaining_series = []
    
        for seriesName in seriesToLaunch:
            try:
                _launch_localize(analysisMDH, seriesName)
            except:
                logger.exception('Error launching analysis for %s' % seriesName)
                
                remaining_series.append(seriesName)
                
        if len(remaining_series) > 0:
            logging.debug('%d series were not launched correctly, waiting 20s and retrying' % len(remaining_series))
            time.sleep(20)


    return HttpResponseRedirect('/status/queues/')

