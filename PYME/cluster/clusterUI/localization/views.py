import logging

import django.forms
from django.http import HttpResponseRedirect
from django.shortcuts import render

logger=logging.getLogger(__name__)

import collections
from PYME.localization import MetaDataEdit as mde
#from PYME.cluster import HTTPTaskPusher
from PYME.cluster import HTTPRulePusher, HTTPTaskPusher

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
    from PYME.localization.FitFactories import import_fit_factory
    #params = FINDING_PARAMS + DEFAULT_PARAMS
    analysisModule=str(analysisModule)

    categorized_fields = collections.OrderedDict()

    categorized_fields['Detection'] = [p.formField() for p in FINDING_PARAMS]
    categorized_fields['Background'] = [p.formField() for p in BACKGROUND_PARAMS]
    categorized_fields['sCMOS'] = [p.formField() for p in SCMOS_PARAMS]
    categorized_fields['Fiducials'] = [p.formField() for p in FIDUCIAL_PARAMS]


    try:
        fm = import_fit_factory(analysisModule)

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
    analysisModule=str(analysisModule)

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


def localize(request, analysisModule='LatGaussFitFR'):
    #import json
    from PYME.IO import MetaDataHandler
    import copy
    import time
    from PYME import config
    USE_RULES = config.get('PYMERuleserver-use', True)

    analysisModule = request.POST.get('Analysis.FitModule', analysisModule)

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
    
    nSeries = len(remaining_series)
    
    nAttempts = 0

    if USE_RULES:
        import posixpath
        from PYME.cluster.rules import LocalisationRuleFactory
        rule_factory = LocalisationRuleFactory(analysisMetadata=analysisMDH)
    
    while len(remaining_series) > 0 and nAttempts < 3:
        nAttempts += 1
        
        seriesToLaunch = copy.copy(remaining_series)
        remaining_series = []
    
        for seriesName in seriesToLaunch:
            try:
                if USE_RULES:
                    context = {'seriesName': seriesName}
                    rule_factory.get_rule(context=context).push()
                else:
                    HTTPTaskPusher.launch_localize(analysisMDH, seriesName)
            except:
                logger.exception('Error launching analysis for %s' % seriesName)
                
                remaining_series.append(seriesName)
                
        if len(remaining_series) > 0:
            logging.debug('%d series were not launched correctly, waiting 20s and retrying' % len(remaining_series))
            time.sleep(20)

    nFailed = len(remaining_series)
    if nFailed > 0:
        raise RuntimeError('Failed to push %d of %d series' % (nFailed, nSeries))

    return HttpResponseRedirect('/status/queues/')
