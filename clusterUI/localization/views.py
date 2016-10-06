from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from django.core.files.uploadhandler import TemporaryFileUploadHandler
from django.views.decorators.csrf import csrf_exempt, csrf_protect
import django.forms


from PYME.localization import MetaDataEdit as mde
FINDING_PARAMS = [mde.FloatParam('Analysis.DetectionThreshold', 'Thresh:', 1.0),
                  mde.IntParam('Analysis.DebounceRadius', 'Debounce rad:', 4),
                  ]


DEFAULT_PARAMS = [mde.IntParam('Analysis.StartAt', 'Start at:', default=30),
              mde.RangeParam('Analysis.BGRange', 'Background:', default=[-30, 0]),
              mde.BoolParam('Analysis.subtractBackground', 'Subtract background in fit', default=True),
              mde.BoolFloatParam('Analysis.PCTBackground', 'Use percentile for background', default=False,
                                 helpText='', ondefault=0.25, offvalue=0),
              mde.FilenameParam('Camera.VarianceMapID', 'Variance Map:',
                                prompt='Please select variance map to use ...', wildcard='TIFF Files|*.tif',
                                filename=''),
              mde.FilenameParam('Camera.DarkMapID', 'Dark Map:', prompt='Please select dark map to use ...',
                                wildcard='TIFF Files|*.tif', filename=''),
              mde.FilenameParam('Camera.FlatfieldMapID', 'Flatfield Map:',
                                prompt='Please select flatfield map to use ...', wildcard='TIFF Files|*.tif',
                                filename=''),
              mde.BoolParam('Analysis.TrackFiducials', 'Track Fiducials', default=False),
              mde.FloatParam('Analysis.FiducialThreshold', 'Fiducial Threshold', default=1.8),
              ]



def settings_form(analysisModule):
    params = FINDING_PARAMS + DEFAULT_PARAMS

    try:
        fm = __import__('PYME.localization.FitFactories.' + analysisModule,
                        fromlist=['PYME', 'localization', 'FitFactories'])

        params += fm.PARAMETERS
    except AttributeError:
        pass

    fields = {}
    for p in params:
        fields.update(p.formField())

    return type('%sForm' % analysisModule, (django.forms.Form, ), fields)


# Create your views here.
def settings(request, analysisModule='LatGaussFitFR'):
    import PYME.localization.FitFactories

    if request.method == 'POST':
        f = settings_form(analysisModule)(request.POST)
    else:
        f = settings_form(analysisModule)()


    return render(request, 'localization/settings_form.html', {'form':f, 'analysisModule' : analysisModule, 'analysisModuleChoices' : PYME.localization.FitFactories.resFitFactories})


def localize(request, analysisModule='LatGaussFitFR', files=[]):
    import json
    f = settings_form(analysisModule)(request.POST)

    f.is_valid()

    f.cleaned_data['Analysis.FitModule'] = analysisModule

    print json.dumps(f.cleaned_data)


    return HttpResponseRedirect('/status/queues/')

