#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################


from django.http import HttpResponse
from samples import models
import json

def JsonResponse(data):
    return HttpResponse(json.dumps(data), content_type="application/json")

def __get_matching_slides(request):
    try:    
        structure = request.REQUEST['structure']
    except KeyError:
        structure = ''
        
    try:    
        creator = request.REQUEST['creator']
    except KeyError:
        creator = ''

    try:    
        reference = request.REQUEST['reference']
    except KeyError:
        reference = ''
    
    if not structure == '':
        qs = models.Slide.objects.filter(creator__contains=creator, reference__contains=reference, labelling__structure__contains=structure).order_by('-timestamp')
    else:
        qs = models.Slide.objects.filter(creator__contains=creator, reference__contains=reference).order_by('-timestamp')

    return qs

def num_matching_slides(request):
    qs = __get_matching_slides(request)
        
    return JsonResponse({'num_matches' : qs.count()})

def get_slide_info(request):
    qs = __get_matching_slides(request)
    
    index = int(request.REQUEST['index'])

    sl = qs[index]

    sl_info = {'slideID' : sl.slideID,
                'creator' : sl.creator,
                'reference' : sl.reference,
                'notes' : sl.notes,
                'sample' : str(sl.sample),
                'labels' : sl.label_list()
                }

    return JsonResponse({'desc' : sl.desc(), 'info' : sl_info})


def get_creator_choices(request):
    slref = request.REQUEST['slref']
    cname = request.REQUEST['cname']

    if slref == '' or (models.Slide.objects.filter(reference=slref).count() ==0):
        choices = list(set([e.creator for e in models.Slide.objects.filter(creator__startswith=cname)]))
    else:
        choices = list(set([e.creator for e in models.Slide.objects.filter(creator__startswith=cname, reference=slref)]))

    return JsonResponse(choices)



def get_slide_choices(request):
    slref = request.REQUEST['slref']
    cname = request.REQUEST['cname']

    if cname == '' or (models.Slide.objects.filter(creator=cname).count() == 0):
        choices = list(set([e.reference for e in models.Slide.objects.filter(reference__startswith=slref)]))
    else:
        choices = list(set([e.reference for e in models.Slide.objects.filter(reference__startswith=slref, creator=cname)]))

    return JsonResponse(choices)


def get_structure_choices(request):
    sname = request.REQUEST['sname']

    choices = list(set([e.structure for e in models.Labelling.objects.filter(structure__startswith=sname)]))

    return JsonResponse(choices)


def get_dye_choices(request):
    dname = request.REQUEST['dname']

    choices = list(set([e.label for e in models.Labelling.objects.filter(label__startswith=dname)]))

    return JsonResponse(choices)

